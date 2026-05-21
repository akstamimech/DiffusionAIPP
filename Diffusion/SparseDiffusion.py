import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from torch import dtype, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import CubicSpline



SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
PLOT_DIR = SCRIPT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

"""
- x_0: one sample is shaped (B, 2, 8) [batch, coords, control waypoint columns]
- data: CMAES_classic_betasweep_dataset.pt
- control waypoints are normalized to [-1, 1]
"""

EPOCHS = 3000
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_COORDS = 2
NUM_CONTROL_WAYPOINTS = 8
TARGET_SHAPE = (NUM_COORDS, NUM_CONTROL_WAYPOINTS)
FLAT_TRAJ_DIM = NUM_COORDS * NUM_CONTROL_WAYPOINTS
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
MIN_LR = 1e-5
INDEX = -1



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
    SCRIPT_DIR / "CMAES_beamsearch_dataset.pt"
)
dense_trajectories = data_dict["trajectories"].float()
control_waypoints = data_dict["control_waypoints"].float()
raw_current_positions = data_dict["current_position"].float()
conditions = raw_current_positions.clone()
means = data_dict["current_mean"].float()
vars = data_dict["current_var"].float()
rmsedrop = data_dict["RMSE_correction"].float()
map_ids = data_dict["map_id"].long()
timesteps = data_dict["timestep"].long()


weighting_utilities = rmsedrop.clamp_min(0.0) + 1e-6
weights = torch.zeros_like(weighting_utilities)

groups = torch.stack([map_ids, timesteps], dim=1)

for group in torch.unique(groups, dim=0):
    group_mask = (groups == group).all(dim=1)
    group_rmsedrop = weighting_utilities[group_mask]
    denom = group_rmsedrop.sum() + 1e-6

    weights[group_mask] = group_rmsedrop / denom




SCALE_FACTOR = 100.0
conditions = (conditions / SCALE_FACTOR) * 2.0 - 1.0

######NORMALIZING AND INDEXING######
if control_waypoints.ndim != 3:
    raise ValueError(f"Expected control_waypoints to be rank 3, got {control_waypoints.shape}")

if control_waypoints.shape[1:] == TARGET_SHAPE:
    trajectories = control_waypoints
elif control_waypoints.shape[1:] == (NUM_CONTROL_WAYPOINTS, NUM_COORDS):
    trajectories = control_waypoints.permute(0, 2, 1)
else:
    raise ValueError(
        "Expected control_waypoints shape [B, 2, 8] or [B, 8, 2], "
        f"got {tuple(control_waypoints.shape)}"
    )

trajectories = (trajectories / SCALE_FACTOR) * 2.0 - 1.0

if INDEX is not None and INDEX > 0:
    trajectories = trajectories[:INDEX]
    conditions = conditions[:INDEX]
    means = means[:INDEX]
    vars = vars[:INDEX]
    rmsedrop = rmsedrop[:INDEX]
    weights = weights[:INDEX]

trajectories = trajectories.to(device)
conditions = conditions.to(device)
means = means.to(device)
vars = vars.to(device)
rmsedrop = rmsedrop.to(device)
weights = weights.to(device)


# conditions = conditions[:1].repeat(INDEX, 1).to(device)
# means = means[:1].repeat(INDEX, 1, 1).to(device)
# vars = vars[:1].repeat(INDEX, 1, 1).to(device)


def denormalize_control_waypoints(waypoints):
    return ((waypoints + 1.0) / 2.0) * SCALE_FACTOR


def extract_control_waypoints(waypoint_tensor):
    return denormalize_control_waypoints(waypoint_tensor)


def pytorch_cubic_spline(control_waypoints, current_position, samples_per_segment=5, eps=1e-9):
    """
    Batched differentiable natural cubic spline.

    Args:
        control_waypoints:
            [B, K, 2] or [B, 2, K] or [K, 2] or [2, K]

        current_position:
            [B, 2] or [2] or flattened [2B]

    Returns:
        dense_path:
            [B, 2, samples_per_segment * K + 1]
    """

    if not torch.is_tensor(control_waypoints):
        control_waypoints = torch.tensor(control_waypoints, dtype=torch.float32)

    device = control_waypoints.device
    dtype = control_waypoints.dtype

    if not torch.is_tensor(current_position):
        current_position = torch.tensor(current_position, dtype=dtype, device=device)

    current_position = current_position.to(device=device, dtype=dtype)

    # -------------------------
    # Normalize control_waypoints to [B, K, 2]
    # -------------------------
    if control_waypoints.ndim == 2:
        # Single sample: [K, 2] or [2, K]
        if control_waypoints.shape[0] == 2 and control_waypoints.shape[1] != 2:
            control_waypoints = control_waypoints.T
        control_waypoints = control_waypoints.unsqueeze(0)  # [1, K, 2]

    elif control_waypoints.ndim == 3:
        # Batched: [B, 2, K] -> [B, K, 2]
        if control_waypoints.shape[1] == 2 and control_waypoints.shape[2] != 2:
            control_waypoints = control_waypoints.transpose(1, 2)

    else:
        raise ValueError(f"control_waypoints must have 2 or 3 dims, got {control_waypoints.shape}")

    B, K, D = control_waypoints.shape

    if D != 2:
        raise ValueError(f"Expected waypoint dimension 2, got shape {control_waypoints.shape}")

    # -------------------------
    # Normalize current_position to [B, 1, 2]
    # -------------------------
    if current_position.ndim == 1:
        if current_position.numel() == 2:
            current_position = current_position.reshape(1, 2).repeat(B, 1)
        elif current_position.numel() == 2 * B:
            current_position = current_position.reshape(B, 2)
        else:
            raise ValueError(
                f"current_position has {current_position.numel()} elements, "
                f"expected 2 or {2 * B}"
            )

    elif current_position.ndim == 2:
        if current_position.shape == (1, 2) and B > 1:
            current_position = current_position.repeat(B, 1)
        elif current_position.shape != (B, 2):
            raise ValueError(
                f"current_position shape {current_position.shape} does not match batch size {B}"
            )

    else:
        raise ValueError(f"current_position must have shape [2], [B,2], or [2B], got {current_position.shape}")

    current_position = current_position.reshape(B, 1, 2)

    # Full waypoint sequence: [B, N, 2], N = K + 1
    waypoints = torch.cat([current_position, control_waypoints], dim=1)

    N = waypoints.shape[1]

    # Segment lengths: [B, N-1]
    deltas = waypoints[:, 1:, :] - waypoints[:, :-1, :]
    h = torch.linalg.norm(deltas, dim=-1).clamp_min(eps)

    # Build batched linear systems A M = rhs
    A = torch.zeros(B, N, N, device=device, dtype=dtype)
    rhs = torch.zeros(B, N, 2, device=device, dtype=dtype)

    # Natural boundary conditions
    A[:, 0, 0] = 1.0
    A[:, -1, -1] = 1.0

    for i in range(1, N - 1):
        h_prev = h[:, i - 1]
        h_next = h[:, i]

        A[:, i, i - 1] = h_prev
        A[:, i, i] = 2.0 * (h_prev + h_next)
        A[:, i, i + 1] = h_next

        slope_next = (waypoints[:, i + 1, :] - waypoints[:, i, :]) / h_next[:, None]
        slope_prev = (waypoints[:, i, :] - waypoints[:, i - 1, :]) / h_prev[:, None]

        rhs[:, i, :] = 6.0 * (slope_next - slope_prev)

    # Second derivatives at knots: [B, N, 2]
    M = torch.linalg.solve(A, rhs)

    trajectory_segments = []

    alphas = torch.linspace(
        0.0,
        1.0,
        samples_per_segment + 1,
        device=device,
        dtype=dtype,
    )[:-1]  # [S_seg]

    for i in range(N - 1):
        hi = h[:, i]  # [B]

        tau = alphas[None, :] * hi[:, None]  # [B, S_seg]

        Acoef = hi[:, None] - tau
        Bcoef = tau

        yi = waypoints[:, i, :]        # [B, 2]
        yi1 = waypoints[:, i + 1, :]   # [B, 2]
        Mi = M[:, i, :]                # [B, 2]
        Mi1 = M[:, i + 1, :]           # [B, 2]

        hi_exp = hi[:, None, None]
        Acoef_exp = Acoef[:, :, None]
        Bcoef_exp = Bcoef[:, :, None]

        points = (
            Mi[:, None, :] * Acoef_exp**3 / (6.0 * hi_exp)
            + Mi1[:, None, :] * Bcoef_exp**3 / (6.0 * hi_exp)
            + (yi - Mi * hi[:, None]**2 / 6.0)[:, None, :] * (Acoef_exp / hi_exp)
            + (yi1 - Mi1 * hi[:, None]**2 / 6.0)[:, None, :] * (Bcoef_exp / hi_exp)
        )

        trajectory_segments.append(points)  # [B, samples_per_segment, 2]

    # Append final waypoint exactly
    trajectory_segments.append(waypoints[:, -1:, :])  # [B, 1, 2]

    dense_path = torch.cat(trajectory_segments, dim=1)  # [B, S, 2]

    return dense_path.transpose(1, 2)  # [B, 2, S]





# def spline_regeneration(sparse_states_generated, samples_per_segment=5):
#     """
#     sparse_states_generated: [2, 9] in world coordinates.
#         column 0 = current position
#         columns 1: = 8 control waypoints

#     returns:
#         dense_path: [2, 41]
#         dense_actions: [2, 40]
#     """
#     waypoints = sparse_states_generated.T  # [9, 2]

#     deltas = np.diff(waypoints, axis=0)
#     seg_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
#     t = np.concatenate([[0.0], np.cumsum(seg_lengths)])

#     keep = np.concatenate([[True], np.diff(t) > 1e-9])
#     waypoints = waypoints[keep]
#     t = t[keep]

#     if len(t) < 2:
#         dense_path = np.repeat(waypoints[:1], 41, axis=0)
#         dense_path = dense_path.T
#         dense_actions = dense_path[:, 1:] - dense_path[:, :-1]
#         return dense_path, dense_actions

#     cs_x = CubicSpline(t, waypoints[:, 0], bc_type="natural")
#     cs_y = CubicSpline(t, waypoints[:, 1], bc_type="natural")

#     trajectory = []
#     for i in range(len(t) - 1):
#         t_segment = np.linspace(t[i], t[i + 1], samples_per_segment, endpoint=False)
#         for ts in t_segment:
#             trajectory.append((float(cs_x(ts)), float(cs_y(ts))))

#     trajectory.append((float(waypoints[-1, 0]), float(waypoints[-1, 1])))

#     dense_path = np.asarray(trajectory, dtype=np.float32).T  # [2, 41]
#     dense_actions = dense_path[:, 1:] - dense_path[:, :-1]  # [2, 40]

#     return dense_path, dense_actions

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
    def __init__(self, trajectories, weights, meanvarmaps = meanvarmaps, conditions=None):
        self.trajectories = trajectories
        self.weights = weights
        self.rmsedrop = rmsedrop
        self.conditions = conditions
        self.meanvarmaps = meanvarmaps

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        if self.conditions is None:
            return self.trajectories[idx]
        return self.trajectories[idx], self.conditions[idx], self.meanvarmaps[idx], self.weights[idx]


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
        self.pool3 = nn.AdaptiveAvgPool2d((12, 12))
        self.flatten = nn.Flatten()
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, pos_hidden_dim),
            nn.SiLU(),
            nn.Linear(pos_hidden_dim, pos_hidden_dim),
            nn.SiLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 12 * 12 + pos_hidden_dim, 256),
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


class ConditionalResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, groups=8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.cond_layer = nn.Linear(cond_dim, out_channels * 2)
        self.act = nn.SiLU()
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        residual = self.residual(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        gamma, beta = self.cond_layer(cond).chunk(2, dim=-1)
        x = (1.0 + gamma[:, :, None]) * x + beta[:, :, None]

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)

        return x + residual


class NoisePredictor(nn.Module):
    def __init__(self, time_emb_dim=64, cond_dim=256, base_channels=64):
        super().__init__()
        input_channels = NUM_COORDS
        self.time_embedding = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )
        self.mean_var_cnn = MeanVarCNN(
            input_channels=2,
            hidden_dim=32,
            output_dim=cond_dim,
            pos_hidden_dim=32,
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.input_proj = nn.Conv1d(input_channels, base_channels, kernel_size=3, padding=1)
        self.down_block = ConditionalResBlock1D(base_channels, base_channels, cond_dim)
        self.downsample = nn.Conv1d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.mid_block = ConditionalResBlock1D(base_channels * 2, base_channels * 2, cond_dim)
        self.upsample = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.up_block = ConditionalResBlock1D(base_channels * 2, base_channels, cond_dim)
        self.output = nn.Conv1d(base_channels, input_channels, kernel_size=1)

    def forward(self, x, t, meanvar_map, current_position):
        batch_size = x.shape[0]
        time_emb = self.time_mlp(self.time_embedding(t))
        meanvar_emb = self.mean_var_cnn(meanvar_map, current_position)
        cond = self.cond_mlp(torch.cat([time_emb, meanvar_emb], dim=-1))

        x = self.input_proj(x)

        skip = self.down_block(x, cond)
        x = self.downsample(skip)
        x = self.mid_block(x, cond)
        x = self.upsample(x)

        if x.shape[-1] < skip.shape[-1]:
            x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
        elif x.shape[-1] > skip.shape[-1]:
            x = x[:, :, :skip.shape[-1]]

        x = torch.cat([x, skip], dim=1)
        x = self.up_block(x, cond)
        x = self.output(x)
        return x.reshape(batch_size, *TARGET_SHAPE)


model = NoisePredictor().to(device)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


def get_loss(model, x_0, t, meanvar_map, current_position, weights, alpha=0.1):
    waypoints_noisy, noise = forward_diffusion_sample(x_0, t) 
    noise_pred = model(waypoints_noisy, t, meanvar_map, current_position)

    waypoint_loss = (noise_pred - noise).pow(2).mean(dim=[1, 2])


    traj_noise_pred = pytorch_cubic_spline(noise_pred, current_position)
    traj_noise_true = pytorch_cubic_spline(noise, current_position)


    spline_loss = (traj_noise_pred - traj_noise_true).pow(2).mean(dim=[1, 2])

    per_sample_loss = waypoint_loss + alpha * spline_loss
    return (per_sample_loss * weights).sum() / (weights.sum() + 1e-6)


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
def ddim_sample_timestep(x, t, t_prev, meanvar_map, current_position, clip_x0=False):
    alpha_bar_t = get_index_from_list(alphas_cumprod, t, x.shape)
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

    noise_pred = model(x, t, meanvar_map, current_position)
    x0_pred = (x - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
    if clip_x0:
        x0_pred = x0_pred.clamp(-1.0, 1.0)

    if t_prev[0].item() < 0:
        return x0_pred

    alpha_bar_prev = get_index_from_list(alphas_cumprod, t_prev, x.shape)
    sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
    sqrt_one_minus_alpha_bar_prev = torch.sqrt(1.0 - alpha_bar_prev)

    x_prev = sqrt_alpha_bar_prev * x0_pred + sqrt_one_minus_alpha_bar_prev * noise_pred
    return x_prev

@torch.no_grad()
def ddim_sample(initial_noise, meanvar_map, current_position, num_steps=None, clip_x0=False):
    x = initial_noise
    if num_steps is None or num_steps >= T:
        schedule = list(range(T - 1, -1, -1))
    else:
        schedule = torch.linspace(T - 1, 0, steps=num_steps, device=x.device).long().unique_consecutive().tolist()

    for step_idx, i in enumerate(schedule):
        prev_i = schedule[step_idx + 1] if step_idx + 1 < len(schedule) else -1
        t = torch.full((x.shape[0],), i, dtype=torch.long, device=x.device)
        t_prev = torch.full((x.shape[0],), prev_i, dtype=torch.long, device=x.device)
        x = ddim_sample_timestep(x, t, t_prev, meanvar_map, current_position, clip_x0=clip_x0)
    return x


@torch.no_grad()
def sample_plot_traj(output_path=None):
    model.eval()
    traj = torch.randn((1, *TARGET_SHAPE), device=next(model.parameters()).device)
    meanvar_map = meanvarmaps[0:1].to(next(model.parameters()).device)
    current_position = conditions[0:1].to(next(model.parameters()).device)
    plt.figure(figsize=(6, 6))
    plt.axis("equal")
    plt.grid(True)

    traj = ddim_sample(traj, meanvar_map, current_position)

    traj_to_plot = extract_control_waypoints(traj[0].cpu())
    #.clamp(0.0, SCALE_FACTOR)
    truth_to_plot = extract_control_waypoints(trajectories[0].cpu())
    plt.plot(traj_to_plot[0], traj_to_plot[1], marker="o", label="Generated control waypoints")
    plt.plot(truth_to_plot[0], truth_to_plot[1], marker="x", label="Ground truth control waypoints")
    plt.legend()
    if output_path is None:
        output_path = PLOT_DIR / "sparse_sample_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved sample plot to {output_path}")



def train_one_sample(model, steps=3000, batch_size=64):
    losses = []
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    model.train()

    x0 = trajectories[:1].to(device)
    pos0 = conditions[:1].to(device)
    map0 = meanvarmaps[:1].to(device)

    for step in range(steps):
        traj = x0.repeat(batch_size, 1, 1)
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

    final_path = CHECKPOINT_DIR / "sparse_control_one_sample_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Saved one-sample model to {final_path}")

    plt.figure()
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("One-sample SD-DA MLP diffusion loss")
    loss_plot_path = PLOT_DIR / "sparse_one_sample_loss.png"
    plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved one-sample loss plot to {loss_plot_path}")

    sample_plot_traj(PLOT_DIR / "sparse_one_sample_sample.png")
    return losses


def train(model, dataloader, epochs, betas=betas, lr=1e-3, save_every=10):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    loss_vals = []
    stepcount = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            stepcount.append(epoch * len(dataloader) + step)
            traj, current_position, meanvar_map, batch_weights = batch
            traj = traj.to(next(model.parameters()).device)
            current_position = current_position.to(next(model.parameters()).device)
            meanvar_map = meanvar_map.to(next(model.parameters()).device)
            batch_size = traj.shape[0]
            t = torch.randint(0, T, (batch_size,), device=traj.device).long()
            loss = get_loss(model, traj, t, meanvar_map, current_position, weights=batch_weights)
            loss_vals.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
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
            checkpoint_path = CHECKPOINT_DIR / f"sparse_waypoints_epoch_{epoch + 1}.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        print(f"Epoch {epoch + 1} complete, lr={lr:.6f}", flush=True)
    
    plt.figure()
    plt.plot(loss_vals, label="Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("SD-DA MLP Diffusion Training Loss")
    plt.legend()
    loss_plot_path = PLOT_DIR / "sparse_training_loss.png"
    plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training loss plot to {loss_plot_path}")

    sample_plot_traj(PLOT_DIR / "sparse_training_sample.png")


   


if __name__ == "__main__":
    print("Checkpoint directory made", flush=True)
    print("Using device:", device, flush=True)

    print("Building MLP diffusion model", flush=True)
    model = NoisePredictor().to(device)

    print("Training one-sample overfit model", flush=True)
    #batch_wreights = torch.ones(batch_size, device=device)
    # train_one_sample(model, steps=EPOCHS, batch_size=BATCH_SIZE)
    train(model, DataLoader(TrajectoryDataset(trajectories, weights, meanvarmaps=meanvarmaps, conditions=conditions), batch_size=BATCH_SIZE, shuffle=True), epochs=EPOCHS)

    print("Done training", flush=True)

    print("control_waypoints shape:", control_waypoints.shape, flush=True)

    
