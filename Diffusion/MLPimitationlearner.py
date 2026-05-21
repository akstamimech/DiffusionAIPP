import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from tqdm import tqdm


os.makedirs("checkpoints", exist_ok=True)

"""
- x_0: one sample is shaped (B, 2, 10) [batch, coords, horizon]
- data: trajectory_dataset.pt
- trajectories are normalized to [0, 1] to keep diffusion noise scales stable
- MLP architecture directly outputting waypoint trajectories instead of noise prediction in this file for comparison with diffusion

"""

EPOCHS = 20
BATCH_SIZE = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAJ_SIZE = 9
NUM_COORDS = 2
FLAT_TRAJ_DIM = NUM_COORDS * TRAJ_SIZE
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
MIN_LR = 1e-5




data_dict = torch.load(
    rf"C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts\Diffusion\trajectory_dataset.pt"
)
trajectories = data_dict["trajectories"].float()
conditions = data_dict["current_position"].float()
means = data_dict["current_mean"].float()
vars = data_dict["current_var"].float()

SCALE_FACTOR = 100.0
trajectories = trajectories / SCALE_FACTOR
conditions = conditions / SCALE_FACTOR

# The conditioning channels have very different scales. Standardizing the mean
# map and log-variance map makes optimization much easier for the MLP.
log_vars = torch.log1p(vars)
mean_center = means.mean()
mean_scale = means.std().clamp_min(1e-6)
log_var_center = log_vars.mean()
log_var_scale = log_vars.std().clamp_min(1e-6)

means = (means - mean_center) / mean_scale
log_vars = (log_vars - log_var_center) / log_var_scale
meanvarmaps = torch.stack([means, log_vars], dim=1)  # shape (B, 2, 51, 51)


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, meanvarmaps = meanvarmaps, conditions=None):
        self.trajectories = trajectories
        self.conditions = conditions
        self.meanvarmaps = meanvarmaps

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        if self.conditions is None:
            return self.trajectories[idx]
        return self.trajectories[idx], self.conditions[idx], self.meanvarmaps[idx]


# class SinusoidalPositionEmbeddings(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, time):
#         device = time.device
#         half_dim = self.dim // 2
#         embeddings = math.log(10000) / max(half_dim - 1, 1)
#         embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
#         embeddings = (time.float() / (T - 1))[:, None] * embeddings[None, :]
#         embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
#         if self.dim % 2 == 1:
#             embeddings = F.pad(embeddings, (0, 1))
#         return embeddings
 
 
class WaypointMLP(nn.Module):
    def __init__(self, input_dim=FLAT_TRAJ_DIM, hidden_dim=256, output_dim=128, num_layers=2):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MeanVarCNN(nn.Module):
    def __init__(self, input_channels=2, hidden_dim=64, output_dim=128, pos_hidden_dim=32):
        super().__init__()


        """
        input maps: (B, 2, 51, 51) [batch, map data, x, y] (2 channels for mean and var)
        """

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.act1 = nn.SiLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1)
        self.act2 = nn.SiLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=hidden_dim * 2, out_channels=hidden_dim * 2, kernel_size=3, padding=1)
        self.act3 = nn.SiLU()
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, pos_hidden_dim),
            nn.SiLU(),
            nn.Linear(pos_hidden_dim, pos_hidden_dim),
            nn.SiLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 4 * 4 + pos_hidden_dim, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, output_dim),
        )

    def forward(self, x, current_position):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.pool3(self.act3(self.conv3(x)))
        x = self.flatten(x)
        pos_emb = self.pos_mlp(current_position)
        x = torch.cat([x, pos_emb], dim=-1)
        x = self.fc(x)
        return x


class TrajPredictor(nn.Module):
    def __init__(self, hidden_dim=512, num_layers=4, traj_emb_dim=128):
        super().__init__()
        self.mean_var_cnn = MeanVarCNN(input_channels=2, hidden_dim=64, output_dim=traj_emb_dim)

        layers = [
            nn.Linear(traj_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        ]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, FLAT_TRAJ_DIM))
        self.traj_mlp = nn.Sequential(*layers)

    def forward(self, meanvar_map, current_position):
        batch_size = meanvar_map.shape[0]
        meanvar_emb = self.mean_var_cnn(meanvar_map, current_position)
        pred_flat = self.traj_mlp(meanvar_emb)
        return pred_flat.view(batch_size, NUM_COORDS, TRAJ_SIZE)


model = TrajPredictor().to(device)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


def get_loss(model, x_0, meanvar_map, current_position):
    x_pred = model(meanvar_map, current_position)
    return F.mse_loss(x_pred, x_0)


# @torch.no_grad()
# def sample_timestep(x, t, meanvar_map, current_position):
#     betas_t = get_index_from_list(betas, t, x.shape)
#     sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
#     sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

#     model_mean = sqrt_recip_alphas_t * (
#         x - betas_t * model(x, t, meanvar_map, current_position) / sqrt_one_minus_alphas_cumprod_t
#     )
#     posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

#     if t[0].item() == 0:
#         return model_mean

#     noise = torch.randn_like(x)
#     return model_mean + torch.sqrt(posterior_variance_t) * noise


# @torch.no_grad()
# def sample_plot_traj():
#     model.eval()
#     traj = torch.randn((1, NUM_COORDS, TRAJ_SIZE), device=next(model.parameters()).device)
#     meanvar_map = meanvarmaps[0:1].to(next(model.parameters()).device)
#     current_position = (conditions[0:1] / SCALE_FACTOR).to(next(model.parameters()).device)
#     plt.figure(figsize=(6, 6))
#     plt.axis("equal")
#     plt.grid(True)

#     for i in range(T - 1, -1, -1):
#         t = torch.full((1,), i, dtype=torch.long, device=device)
#         traj = sample_timestep(traj, t, meanvar_map, current_position)

#     traj_to_plot = (traj[0].cpu() * SCALE_FACTOR)
#     #.clamp(0.0, SCALE_FACTOR)
#     plt.plot(traj_to_plot[0], traj_to_plot[1], marker="o")
#     plt.show()


def train(model, dataloader, epochs, lr=LR, save_every=10):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=MIN_LR)
    model.train()
    loss_vals = []
    stepcount = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            stepcount.append(epoch * len(dataloader) + step)
            traj, current_position, meanvar_map = batch
            traj = traj.to(next(model.parameters()).device)
            current_position = current_position.to(next(model.parameters()).device)
            meanvar_map = meanvar_map.to(next(model.parameters()).device)
            loss = get_loss(model, traj, meanvar_map, current_position)
            loss_vals.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}", flush=True)

        if (epoch + 1) % save_every == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
            }
            torch.save(checkpoint, f"checkpoints/checkpoint_epoch_{epoch + 1}.pth")
            print(f"Checkpoint saved: checkpoints/checkpoint_epoch_{epoch + 1}.pth")

        scheduler.step()
        print(f"Epoch {epoch + 1} complete, lr={scheduler.get_last_lr()[0]:.6f}", flush=True)

    return loss_vals, stepcount


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    print("Checkpoint directory made", flush=True)
    print("Using device:", device, flush=True)

    print("Building MLP deterministic model", flush=True)
    model = TrajPredictor().to(device)

    print("Loading data", flush=True)
    data = TrajectoryDataset(trajectories, conditions=conditions)
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print("Training model", flush=True)
    loss_vals, stepcount = train(model, dataloader, epochs=EPOCHS, lr=LR, save_every=10)

    torch.save(model.state_dict(), "C:\\Users\\Aksha\\OneDrive\\Year 6\\Thesis\\scripts\\Diffusion\\checkpoints\\mlp_imitation_traj_final.pth")
    print("Final model saved to diffusion/checkpoints/mlp_imitation_traj_final.pth")


    plt.plot(stepcount, loss_vals)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()
    print("Done training", flush=True)

    