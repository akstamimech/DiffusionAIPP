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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-4
NUM_COORDS = 2
TRAJ_SIZE = 40
BATCH_SIZE = 16
EPOCHS = 5000


SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

"""
- x_0: one sample is shaped (B, 2, 41) [batch, coords, horizon]
- data: CMAES_classic_betasweep_dataset.pt
- trajectories are normalized to [0, 1] to keep diffusion noise scales stable
"""



####FORWARD PROCESS FUNCTIONS###

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
    rf"C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts\Diffusion\CMAES_classic_betasweep_dataset.pt"
)
trajectories = data_dict["trajectories"].float()
absolute_trajectories = trajectories.clone()
start_positions = absolute_trajectories[:, :, 0]
trajectories = absolute_trajectories[:, :, 1:] - absolute_trajectories[:, :, :-1]
current_positions = data_dict["current_position"].float()
means = data_dict["current_mean"].float()
vars = data_dict["current_var"].float()

SCALE_FACTOR = 100.0
trajectories = trajectories / SCALE_FACTOR
delta_mean = trajectories.mean()
delta_std = trajectories.std().clamp_min(1e-6)
trajectories = (trajectories - delta_mean) / delta_std
####NOISE PREDICTION MODEL####


def denormalize_delta_trajectory(delta_traj):
    return (delta_traj * delta_std.cpu() + delta_mean.cpu()) * SCALE_FACTOR


def reconstruct_waypoints(delta_traj, start_position):
    delta_world = denormalize_delta_trajectory(delta_traj)
    if start_position.dim() == 1:
        start_world = start_position[:, None]
    else:
        start_world = start_position[:, :, None]
    return start_world + torch.cumsum(delta_world, dim=-1)



##NEED TO CHANGE
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


class TrajectoryDataset(Dataset): ##NEED TO CHANGE
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
 

class TimeResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, kernel_size=5, groups=8):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.time_layer = nn.Linear(time_emb_dim, out_channels)
        self.act = nn.SiLU()
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_emb):
        residual = self.residual(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = x + self.time_layer(time_emb).unsqueeze(-1)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)

        return x + residual


class UNet(nn.Module):
    def __init__(self, trajectory_channels = 2, down_channels = 64, time_emb_dim = 128): 
        super(UNet, self).__init__()

        self.time_embedding = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        self.downblock1 = TimeResBlock(trajectory_channels, down_channels, time_emb_dim)
        self.downblock2 = TimeResBlock(down_channels, down_channels * 2, time_emb_dim)
        self.downblock3 = TimeResBlock(down_channels * 2, down_channels * 4, time_emb_dim)
        self.pool = nn.MaxPool1d(2)

        self.bottleneck = TimeResBlock(down_channels * 4, down_channels * 4, time_emb_dim)

        self.upsample1 = nn.Sequential(
            nn.ConvTranspose1d(down_channels * 4, down_channels * 2, kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.upblock1 = TimeResBlock(down_channels * 6, down_channels * 2, time_emb_dim)

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose1d(down_channels * 2, down_channels, kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.upblock2 = TimeResBlock(down_channels * 3, down_channels, time_emb_dim)

        self.upsample3 = nn.Sequential(
            nn.ConvTranspose1d(down_channels, down_channels, kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.upblock3 = TimeResBlock(down_channels * 2, down_channels, time_emb_dim)
        self.output = nn.Conv1d(down_channels, trajectory_channels, kernel_size=1)

        ##Add attention blocks later if local trajectories look smooth but globally incoherent.

    def forward(self, x, t):
        time_emb = self.time_embedding(t)
        time_emb = self.time_mlp(time_emb)

        res1 = self.downblock1(x, time_emb)
        x = self.pool(res1)
        res2 = self.downblock2(x, time_emb)
        x = self.pool(res2)
        res3 = self.downblock3(x, time_emb)
        x = self.pool(res3)

        x = self.bottleneck(x, time_emb)

        x = self.upsample1(x)
        x = torch.cat([x, res3], dim=1)
        x = self.upblock1(x, time_emb)

        x = self.upsample2(x)
        x = torch.cat([x, res2], dim=1)
        x = self.upblock2(x, time_emb)

        x = self.upsample3(x)
        x = torch.cat([x, res1], dim=1)
        x = self.upblock3(x, time_emb)

        return self.output(x)
        


model = UNet().to(device)
optimizer = AdamW(model.parameters(), lr=LR)



### TRAINING FUNCTIONS AND REVERSE PROCESS######

def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t)
    noise_pred = model(x_noisy, t)
    return F.mse_loss(noise, noise_pred)
    


# @torch.no_grad()
# def sample_timestep(x, t):
#     betas_t = get_index_from_list(betas, t, x.shape)
#     sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
#     sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
#
#     model_mean = sqrt_recip_alphas_t * (
#         x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
#     )
#     posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
#
#     if t[0].item() == 0:
#         return model_mean
#
#     noise = torch.randn_like(x)
#     return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def ddim_sample_timestep(x, t, t_prev):
    alpha_bar_t = get_index_from_list(alphas_cumprod, t, x.shape)
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

    noise_pred = model(x, t)
    x0_pred = (x - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t

    if t_prev[0].item() < 0:
        return x0_pred

    alpha_bar_prev = get_index_from_list(alphas_cumprod, t_prev, x.shape)
    sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
    sqrt_one_minus_alpha_bar_prev = torch.sqrt(1.0 - alpha_bar_prev)

    return sqrt_alpha_bar_prev * x0_pred + sqrt_one_minus_alpha_bar_prev * noise_pred


@torch.no_grad()
def ddim_sample(initial_noise):
    x = initial_noise
    for i in range(T - 1, -1, -1):
        t = torch.full((x.shape[0],), i, dtype=torch.long, device=x.device)
        t_prev = torch.full((x.shape[0],), i - 1, dtype=torch.long, device=x.device)
        x = ddim_sample_timestep(x, t, t_prev)
    return x


@torch.no_grad()
def sample_plot_traj():
    model.eval()
    traj = torch.randn((1, NUM_COORDS, TRAJ_SIZE), device=next(model.parameters()).device)
    # meanvar_map = meanvarmaps[0:1].to(next(model.parameters()).device)
    # current_position = (conditions[0:1] / SCALE_FACTOR).to(next(model.parameters()).device)
    plt.figure(figsize=(6, 6))
    plt.axis("equal")
    plt.grid(True)

    traj = ddim_sample(traj)

    traj_to_plot = reconstruct_waypoints(traj[0].cpu(), start_positions[0].cpu())
    #.clamp(0.0, SCALE_FACTOR)
    plt.plot(traj_to_plot[0], traj_to_plot[1], marker="o", label="Generated Trajectory")
    truth_to_plot = reconstruct_waypoints(trajectories[0].cpu(), start_positions[0].cpu())
    plt.plot(truth_to_plot[0], truth_to_plot[1], marker="x", label="Ground Truth")
    plt.legend()
    plt.show()



def train(model, optimizer, epochs=EPOCHS):
    losses = []
    model.train()
    # dataset = TrajectoryDataset(trajectories) #add meanvarmaps and conditions here
    dataset = TrajectoryDataset(trajectories[:1])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader):
            traj_batch= batch
            traj_batch = traj_batch.to(device)
            # meanvar_batch = meanvar_batch.to(device)
            t = torch.randint(0, T, (traj_batch.size(0),), device=device).long()
            loss = get_loss(model, traj_batch, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss / len(dataloader))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")
        if (epoch + 1) % 10 == 0:
            checkpoint_path = CHECKPOINT_DIR / f"unet_diffusion_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
        
        if (epoch + 1) % epochs == 0:
            plt.plot(losses)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.show()



if __name__ == "__main__":
    losses = train(model, optimizer, epochs=EPOCHS)
    sample_plot_traj()
    
   
