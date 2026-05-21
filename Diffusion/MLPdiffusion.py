import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from pathlib import Path
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

"""
- x_0: one sample is shaped (B, 2, 6, 9)
  [batch, coords, state/action rows, sparse-state columns]
- data: CMAES_classic_betasweep_dataset.pt
- sparse states are normalized to [-1, 1]; dense actions are normalized by SCALE_FACTOR
"""

EPOCHS = 600
BATCH_SIZE = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_COORDS = 2
NUM_CONTROL_WAYPOINTS = 8
NUM_SPARSE_STATES = NUM_CONTROL_WAYPOINTS + 1
ACTIONS_PER_SEGMENT = 5
SDDA_ROWS = ACTIONS_PER_SEGMENT + 1
TARGET_SHAPE = (NUM_COORDS, SDDA_ROWS, NUM_SPARSE_STATES)
FLAT_TRAJ_DIM = NUM_COORDS * SDDA_ROWS * NUM_SPARSE_STATES
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
MIN_LR = 1e-5


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos((((x / timesteps) + s) / (1 + s)) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-4, 0.05)


def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t):
    noise = torch.randn_like(x_0)

    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)

    x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    return x_t, noise


T = 1000
betas = cosine_beta_schedule(timesteps=T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)


data_dict = torch.load(
    SCRIPT_DIR / "CMAES_classic_betasweep_dataset.pt"
)
dense_trajectories = data_dict["trajectories"].float()
control_waypoints = data_dict["control_waypoints"].float()
conditions = data_dict["current_position"].float()
means = data_dict["current_mean"].float()
vars = data_dict["current_var"].float()

SCALE_FACTOR = 100.0
conditions = conditions / SCALE_FACTOR


def build_sparse_dense_action_tensor(dense_paths, controls, current_positions):
    sparse_states = torch.cat([current_positions[:, None, :], controls], dim=1)
    dense_actions = dense_paths[:, :, 1:] - dense_paths[:, :, :-1]
    dense_actions = dense_actions.permute(0, 2, 1).reshape(
        dense_paths.shape[0],
        NUM_CONTROL_WAYPOINTS,
        ACTIONS_PER_SEGMENT,
        NUM_COORDS,
    )

    padded_actions = torch.zeros(
        dense_paths.shape[0],
        NUM_SPARSE_STATES,
        ACTIONS_PER_SEGMENT,
        NUM_COORDS,
        dtype=dense_paths.dtype,
        device=dense_paths.device,
    )
    padded_actions[:, :-1] = dense_actions

    sparse_states = (sparse_states / SCALE_FACTOR) * 2.0 - 1.0
    padded_actions = padded_actions / SCALE_FACTOR

    sdda = torch.cat([sparse_states[:, :, None, :], padded_actions], dim=2)
    return sdda.permute(0, 3, 2, 1).contiguous()


trajectories = build_sparse_dense_action_tensor(
    dense_trajectories,
    control_waypoints,
    data_dict["current_position"].float(),
)


def denormalize_sparse_states(sparse_states):
    return ((sparse_states + 1.0) / 2.0) * SCALE_FACTOR


def extract_sparse_states(sdda_tensor):
    if sdda_tensor.dim() == 4:
        return denormalize_sparse_states(sdda_tensor[:, :, 0, :])
    return denormalize_sparse_states(sdda_tensor[:, 0, :])

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


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / max(half_dim - 1, 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = (time.float() / (T - 1))[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings
 
 
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


class NoisePredictor(nn.Module):
    def __init__(self, time_emb_dim=64, hidden_dim=512, num_layers=4, traj_emb_dim=128):
        super().__init__()
        self.time_embedding = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.traj_encoder = WaypointMLP(
            input_dim=FLAT_TRAJ_DIM,
            hidden_dim=hidden_dim,
            output_dim=traj_emb_dim,
            num_layers=2,
        )

        self.mean_var_cnn = MeanVarCNN(input_channels=2, hidden_dim=64, output_dim=traj_emb_dim)

        layers = [
            nn.Linear(traj_emb_dim + hidden_dim + traj_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        ]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, FLAT_TRAJ_DIM))
        self.noise_mlp = nn.Sequential(*layers)

    def forward(self, x, t, meanvar_map, current_position):
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        traj_emb = self.traj_encoder(x_flat)
        time_emb = self.time_mlp(self.time_embedding(t))
        meanvar_emb = self.mean_var_cnn(meanvar_map, current_position)

        combined = torch.cat([traj_emb, time_emb, meanvar_emb], dim=-1)
        pred_flat = self.noise_mlp(combined)
        return pred_flat.view(batch_size, *TARGET_SHAPE)


model = NoisePredictor().to(device)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


def get_loss(model, x_0, t, meanvar_map, current_position):
    x_noisy, noise = forward_diffusion_sample(x_0, t)
    noise_pred = model(x_noisy, t, meanvar_map, current_position)
    return F.mse_loss(noise, noise_pred)


# @torch.no_grad()
# def sample_timestep(x, t, meanvar_map, current_position):
#     betas_t = get_index_from_list(betas, t, x.shape)
#     sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
#     sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
#
#     model_mean = sqrt_recip_alphas_t * (
#         x - betas_t * model(x, t, meanvar_map, current_position) / sqrt_one_minus_alphas_cumprod_t
#     )
#     posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
#
#     if t[0].item() == 0:
#         return model_mean
#
#     noise = torch.randn_like(x)
#     return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def ddim_sample_timestep(x, t, t_prev, meanvar_map, current_position):
    alpha_bar_t = get_index_from_list(alphas_cumprod, t, x.shape)
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

    noise_pred = model(x, t, meanvar_map, current_position)
    x0_pred = (x - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t

    if t_prev[0].item() < 0:
        return x0_pred

    alpha_bar_prev = get_index_from_list(alphas_cumprod, t_prev, x.shape)
    sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
    sqrt_one_minus_alpha_bar_prev = torch.sqrt(1.0 - alpha_bar_prev)

    return sqrt_alpha_bar_prev * x0_pred + sqrt_one_minus_alpha_bar_prev * noise_pred


@torch.no_grad()
def ddim_sample(initial_noise, meanvar_map, current_position):
    x = initial_noise
    for i in range(T - 1, -1, -1):
        t = torch.full((x.shape[0],), i, dtype=torch.long, device=x.device)
        t_prev = torch.full((x.shape[0],), i - 1, dtype=torch.long, device=x.device)
        x = ddim_sample_timestep(x, t, t_prev, meanvar_map, current_position)
    return x


@torch.no_grad()
def sample_plot_traj():
    model.eval()
    traj = torch.randn((1, *TARGET_SHAPE), device=next(model.parameters()).device)
    meanvar_map = meanvarmaps[0:1].to(next(model.parameters()).device)
    current_position = conditions[0:1].to(next(model.parameters()).device)
    plt.figure(figsize=(6, 6))
    plt.axis("equal")
    plt.grid(True)

    traj = ddim_sample(traj, meanvar_map, current_position)

    traj_to_plot = extract_sparse_states(traj[0].cpu())
    #.clamp(0.0, SCALE_FACTOR)
    truth_to_plot = extract_sparse_states(trajectories[0].cpu())
    plt.plot(traj_to_plot[0], traj_to_plot[1], marker="o", label="Generated sparse states")
    plt.plot(truth_to_plot[0], truth_to_plot[1], marker="x", label="Ground truth sparse states")
    plt.legend()
    plt.show()



def train_one_sample(model, steps=3000, batch_size=64):
    losses = []
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    model.train()

    x0 = trajectories[:1].to(device)
    pos0 = conditions[:1].to(device)
    map0 = meanvarmaps[:1].to(device)

    for step in range(steps):
        traj = x0.repeat(batch_size, 1, 1, 1)
        current_position = pos0.repeat(batch_size, 1)
        meanvar_map = map0.repeat(batch_size, 1, 1, 1)

        t = torch.randint(0, T, (batch_size,), device=device).long()
        loss = get_loss(model, traj, t, meanvar_map, current_position)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"step {step}, loss {loss.item():.4f}")

    final_path = CHECKPOINT_DIR / "mlp_sdda_one_sample_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Saved one-sample model to {final_path}")

    plt.figure()
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("One-sample SD-DA MLP diffusion loss")
    plt.show()

    sample_plot_traj()
    return losses


def train(model, dataloader, epochs, betas=betas, lr=LR, save_every=10):
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
            batch_size = traj.shape[0]
            t = torch.randint(0, T, (batch_size,), device=traj.device).long()
            loss = get_loss(model, traj, t, meanvar_map, current_position)
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
            checkpoint_path = CHECKPOINT_DIR / f"mlp_control_waypoints_epoch_{epoch + 1}.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        scheduler.step()
        print(f"Epoch {epoch + 1} complete, lr={scheduler.get_last_lr()[0]:.6f}", flush=True)

    return loss_vals, stepcount


if __name__ == "__main__":
    print("Checkpoint directory made", flush=True)
    print("Using device:", device, flush=True)

    print("Building MLP diffusion model", flush=True)
    model = NoisePredictor().to(device)

    print("Training one-sample overfit model", flush=True)
    train_one_sample(model, steps=EPOCHS, batch_size=BATCH_SIZE)

    print("Done training", flush=True)

    
