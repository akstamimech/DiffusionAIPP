import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from torch import nn
from torch.optim import Adam
import os
from torchinfo import summary
from torchviz import make_dot
from tqdm import tqdm
from torch.utils.data import Subset

os.makedirs("checkpoints", exist_ok=True)

'''
CONDITIONAL DIFFUSION POLICY
===========================
Based on Chi et al. "Diffusion Policy: Visuomotor Policy Learning through Brain-Computer-Vision Generative Modeling"

Key differences from base_1d.py:
- x_0: shaped (1, 2, 17) [trajectory batch * coords * horizon]
- obs: current position/state that conditions the trajectory generation
- The UNet now takes observations as input to modulate feature generation
- Conditioning is applied via FiLM (Feature-wise Linear Modulation)

Architecture:
- Observation encoder: projects raw observations -> embedding
- UNet blocks: now accept and use observation embeddings to modulate features
- This allows the model to learn: "given this state, what trajectory should I generate?"

OR 
- MLP because the problem is small and we want to test conditioning first without convolutional inductive bias.
- utilities cnn conditioning still feeds into the MLP

'''

#first we define how much param beta change (how much information we change per step)
def linear_beta_schedule(timesteps, start = 0.0001, end = 0.02): 
    #linear schedule
    return torch.linspace(start, end, timesteps)

def linear_beta_schedule_light(timesteps, start=0.00001, end=0.005):
    """Light noise schedule for debugging/single-sample tests"""
    return torch.linspace(start, end, timesteps)
    

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t):  
    noise = torch.randn_like(x_0) #[B, 2, 7]

    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)

    x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    x_t_clamped = torch.clamp(x_t, 0.0, 1.0)
    return x_t_clamped, noise



T = 100
betas = linear_beta_schedule(timesteps=T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)



#Dataloader stuff
TRAJ_SIZE = 18
BATCH_SIZE = 12


data_dict = torch.load(rf"C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts\trajectory_dataset.pt")
trajectories = data_dict["trajectories"].float()
conditions = data_dict["current_position"].float()
utilities = data_dict["utility"].float()

print(trajectories.shape)  # Expecting [N, 2, 18]

# Check utilities shape and add channel dimension if needed
print(f"Utilities shape: {utilities.shape}", flush=True)
if len(utilities.shape) == 3:  # [N, 51, 51] -> [N, 1, 51, 51]
    utilities = utilities.unsqueeze(1)
    print(f"Added channel dimension: {utilities.shape}", flush=True)


# Normalize trajectory coordinates to [0, 1] by dividing by 100.
# This preserves relative differences and makes noise scale meaningful for diffusion.
SCALE_FACTOR = 100.0
trajectories = trajectories / SCALE_FACTOR

# Normalize observations similarly for consistent scale
conditions = conditions / SCALE_FACTOR


class TrajectoryDataset(Dataset): 
    def __init__(self, trajectories, conditions = None, utilities = None): 
        self.trajectories = trajectories
        self.conditions = conditions
        self.utilities = utilities

    def __len__(self): 
        return len(self.trajectories)
    
    def __getitem__(self, idx): 
        if self.conditions is None: 
            return self.trajectories[idx]
        
        else: 
            # Return both trajectory and observation
            return self.trajectories[idx], self.conditions[idx], self.utilities[idx]



data = TrajectoryDataset(trajectories, conditions, utilities)
# data = Subset(data, range(1))  # Use only the first sample for testing
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

#----------------------------------------------UNET ARCHITECTURE WITH CONDITIONAL BLOCKS----------------------------------------------

#U-NET WITH CONDITIONAL BLOCKS
#input: noisy trajectory, timestep t, observation (current state)
#neural networks share params across time so timesteps need to be embedded
#observation conditions the trajectory generation via FiLM (Feature-wise Linear Modulation)
#output: predicted noise


class ConditionalBlock(nn.Module): 
    """
    UNet block that accepts observation conditioning via FiLM.
    
    FiLM (Feature-wise Linear Modulation):
    Instead of just adding time embeddings, we use observations to generate
    scale (gamma) and shift (beta) parameters that modulate intermediate features.
    This allows the network to learn context-dependent denoising.
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, util_emb_dim, up = False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        # FiLM conditioning from observations
        # Generates scale (gamma) and shift (beta) to modulate features: h' = gamma * h + beta
        # self.obs_film_mlp = nn.Sequential(
        #     nn.Linear(obs_emb_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, out_channels * 2)  # [B, out_channels*2]
        # )

        # NEW: FiLM conditioning from utilities grid
        # Also generates scale and shift, applied AFTER observation modulation
        self.util_film_mlp = nn.Sequential(
            nn.Linear(util_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_channels * 2)  # [B, out_channels*2]
        )


        if up:
            # upsample in the decoder path
            self.conv1 = nn.ConvTranspose1d(2*in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.transform = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        else:
            # downsample in the encoder path
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.transform = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        self.conv2 = nn.Conv1d(out_channels, out_channels,  kernel_size= 3, padding= 1) 
        self.relu = nn.ReLU()

    def forward(self, x, t, util_emb):
        # First convolution (down or up sample)
        h = self.relu(self.conv1(x))
        
        # Step 1: Add time embedding
        # time_emb: [B, time_emb_dim] -> [B, out_channels, 1] for broadcasting
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[:, :, None]
        h = h + time_emb  # [B, out_channels, T] += [B, out_channels, 1] broadcasts
        
        # Step 2: Apply observation FiLM modulation
        # obs_emb: [B, obs_emb_dim] -> [B, out_channels*2] -> split into gamma, beta
        # film_params_obs = self.obs_film_mlp(obs_emb)
        # gamma_obs, beta_obs = torch.chunk(film_params_obs, 2, dim=-1)
        # gamma_obs = gamma_obs[:, :, None]  # [B, out_channels, 1]
        # beta_obs = beta_obs[:, :, None]    # [B, out_channels, 1]
        # h = gamma_obs * h + beta_obs
        
        # Step 3: Apply utilities grid FiLM modulation
        # util_emb: [B, util_emb_dim] -> [B, out_channels*2] -> split into gamma, beta
        # This is stacked on top of observation modulation
        film_params_util = self.util_film_mlp(util_emb)
        gamma_util, beta_util = torch.chunk(film_params_util, 2, dim=-1)
        gamma_util = gamma_util[:, :, None]  # [B, out_channels, 1]
        beta_util = beta_util[:, :, None]    # [B, out_channels, 1]
        h = gamma_util * h + beta_util
        
        # Final convolution and transform
        h = self.relu(self.conv2(h))
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalUNET(nn.Module):
    """
    Conditional UNet that generates trajectories conditioned on observations.
    
    Based on Chi et al. "Diffusion Policy" approach:
    - Observations are encoded into an embedding
    - This embedding is used to modulate features via FiLM in each block
    - The network learns: p(trajectory | observation, timestep)
    """

    def __init__(self): 
        super().__init__()
        trajectory_channels = 2  # (x, y)
        down_channels = (16, 32)
        up_channels = down_channels[::-1]
        out_dim = 2
        time_emb_dim = 32 
        # obs_emb_dim = 32  # NEW: embedding dimension for observations

        # NEW: Observation encoder
        # Projects raw observations to a learned embedding space
        # self.obs_encoder = nn.Sequential(
        #     nn.Linear(obs_dim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, obs_emb_dim)
        # )

        #TIME EMBEDDING - we embed the time step t into a vector of size time_emb_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )   

        self.conv0 = nn.Conv1d(trajectory_channels, down_channels[0], kernel_size= 3, padding=1)

        # NEW: Process utilities grid ONCE to get a fixed embedding
        # Input: [B, 1, 51, 51] utilities grid -> Uses convolutions to extract spatial features
        # Output: [B, 32] embedding
        util_emb_dim = 32
        self.util_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=2),  # [B,1,51,51] -> [B,16,26,26]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),  # [B,16,26,26] -> [B,32,13,13]
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),  # [B,32,13,13] -> [B,32,13,13]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # [B,32,13,13] -> [B,32,1,1] global average
            nn.Flatten(1),  # [B,32,1,1] -> [B,32] (keep batch dim, flatten rest)
        )

        # Use ConditionalBlock with utilities embedding dimension
        self.downs = nn.ModuleList([
            ConditionalBlock(down_channels[i], down_channels[i+1], time_emb_dim, util_emb_dim) 
            for i in range(len(down_channels)-1)
        ])

        self.ups = nn.ModuleList([
            ConditionalBlock(up_channels[i], up_channels[i+1], time_emb_dim, util_emb_dim, up = True) 
            for i in range(len(up_channels)-1)
        ]) 

        self.output = nn.Conv1d(up_channels[-1], out_dim, kernel_size= 1)


    def forward(self, x, t, utilities_grid):
        """
        Forward pass with observation and utilities conditioning.
        
        Args:
            x: noisy trajectory [B, 2, T]
            t: timestep indices [B]
            obs: observation/state [B, obs_dim]
            utilities_grid: utility field [B, 1, 51, 51]
        
        Returns:
            predicted noise [B, 2, T]
        """
        # Encode observation to embedding
        # obs_emb = self.obs_encoder(obs)  # [B, obs_emb_dim] = [B, 32]
        
        # Process utilities grid ONCE to get embedding
        # This embedding is reused in all blocks (down and up)
        util_emb = self.util_cnn(utilities_grid)  # [B, util_emb_dim] = [B, 32]
        
        # Embed timestep
        t_emb = self.time_mlp(t)  # [B, time_emb_dim] = [B, 32]
        
        # Initial convolution
        x = self.conv0(x)
        
        # Downsampling with residual connections
        residual_inputs = []
        for down in self.downs:
            # Pass: trajectory features, time embedding, observation embedding, utilities embedding
            x = down(x, t_emb, util_emb)
            residual_inputs.append(x)

        # Upsampling with skip connections
        for up in self.ups:
            skip = residual_inputs.pop()
            # Align time dimension if needed (due to stride=2 in convolutions)
            if x.shape[2] != skip.shape[2]:
                if x.shape[2] < skip.shape[2]:
                    x = F.pad(x, (0, skip.shape[2] - x.shape[2]))
                else:
                    x = x[:, :, :skip.shape[2]]
            x = torch.cat((x, skip), dim=1)
            # Pass all conditions to up blocks too
            x = up(x, t_emb, util_emb)
            
        return self.output(x)
        
#--------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------MLP architecture with conditional blocks-----------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------

class featuremix(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, in_channels)
        )

    def forward(self, x):
        # x: [B, traj, features]
        mixed = self.mlp(x)  # [B, traj, features]
        return x + mixed  # Residual connection
    
class trajmix(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, in_channels)
        )

    def forward(self, x):
        # x: [B, features, traj]
        mixed = self.mlp(x)  # [B, features, traj]
        return x + mixed  # Residual connection


class MLPConditional(nn.Module):
    def __init__(self, time_emb_dim, util_emb_dim): 

        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim), # [B, time_emb_dim]
            nn.Linear(time_emb_dim, time_emb_dim), #[B, time_emb_dim]
            nn.ReLU()) #B, time_emb_dim
        

        self.mlp = nn.Sequential( #[B, 2, traj] -> [B, 32, traj]
            nn.Linear(2, 32), 
            nn.ReLU(),)

        

        self.utilcnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=2),  # [B,1,51,51] -> [B,16,26,26]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # [B,16,26,26] -> [B,16,13,13]
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),  # [B,16,13,13] -> [B,32,7,7]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # [B,32,7,7] -> [B,32,1,1] global average
            nn.Flatten(1),  # [B,32,1,1] -> [B,32] (keep batch dim, flatten rest)
        )

        # self.util_film_mlp = nn.Sequential(
        #     nn.Linear(util_emb_dim, layer_dim),
        #     nn.ReLU(),
        #     nn.Linear(layer_dim, layer_dim * 2)  # [B, 32*2] -> split into gamma, beta
        # )

        self.featuremix = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32)) #mixes features across coordinate dim
        #[1, 32, traj] -> [1, 32, traj]
        self.trajmix = nn.Sequential(nn.Linear(TRAJ_SIZE, 64), nn.ReLU(), nn.Linear(64, TRAJ_SIZE)) #mixes features across trajectory dim
        #[1, 32, traj] -> [1, 32, traj]

        self.out = nn.Linear(32, 2) #final output to predict noise in x and y coords for each point in trajectory



        # util_emb_dim = 32
        self.util_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=2),  # [B,1,51,51] -> [B,16,26,26]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),  # [B,16,26,26] -> [B,32,13,13]
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),  # [B,32,13,13] -> [B,32,13,13]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # [B,32,13,13] -> [B,32,1,1] global average
            nn.Flatten(1),  # [B,32,1,1] -> [B,32] (keep batch dim, flatten rest)
        )


        
        

    def forward(self, x, t):

        x = x.transpose(1, 2)  # [B, 2, traj] -> [B, traj, 2] for linear layer
        t_emb = self.time_mlp(t) # [B, t] --> [B, time_emb_dim]
        traj = self.mlp(x) # [B, 2, traj] -> [B, traj, 32]
        t_emb = t_emb[:, None, :] # [B, time_emb_dim] -> [B, time_emb_dim, 1]
        traj = traj + t_emb  # Broadcast time embedding across coordinate dimension
        #[1, traj, 32]

        #---------time embedded------

        

        traj = traj + self.featuremix(traj) #mix features across coordinate dim
        #[1, traj, 32]
        traj = traj.transpose(1, 2) # [B, traj, 32] -> [B, 32, traj]
        traj = traj + self.trajmix(traj) #mix features across trajectory dim
        #[1, 32, traj]
        traj = traj.transpose(1, 2) # [B, 32, traj] -> [B, traj, 32]
        return self.out(traj).transpose(1, 2) # [B, traj, 32] -> [B, traj, 2] -> [B, 2, traj]
    

        






#LOSS FUNCTION

def get_loss(model, x_0, t, utilities_grid):
    """
    Loss function for conditional diffusion.
    The model predicts noise conditioned on timestep, observation, and utilities.
    """
    x_noisy, noise = forward_diffusion_sample(x_0, t)
    # Pass all conditions: trajectory, time, observation, utilities
    noise_pred = model(x_noisy, t, utilities_grid)
    return F.mse_loss(noise, noise_pred)



#sampling: one denoising step at a time
@torch.no_grad()
def sample_timestep(x, t, utilities_grid): 
    """
    Sampling with observation and utilities conditioning.
    Predicts the noise in the trajectory at timestep t, returns denoised trajectory.
    """

    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Predict noise conditioned on observation and utilities
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t, utilities_grid) / sqrt_one_minus_alphas_cumprod_t)
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t[0].item() == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise
    


@torch.no_grad()
def sample_plot_traj(obs, utilities_grid): 
    """
    Sampling with observation and utilities conditioning.
    Generate a trajectory conditioned on specific observation and utility field.
    """
    model.eval()
    traj_size = TRAJ_SIZE
    traj = torch.randn((1, 2, traj_size)).to(next(model.parameters()).device)
    # obs = obs.to(next(model.parameters()).device).unsqueeze(0)
    utilities_grid = utilities_grid.to(next(model.parameters()).device).unsqueeze(0)  # Add batch dimension
    
    plt.figure(figsize=(15, 15))
    plt.axis("off")
    num_images = 10
    stepsize = int(T / num_images)

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, dtype=torch.long, device=device)
        traj = sample_timestep(traj, t, obs, utilities_grid)
        traj = torch.clamp(traj, -1.0, 1.0)
        
        #initial condition constraint
        traj[:, :, 0] = obs
        
        if i % stepsize == 0:
            print(traj)




#TRAINING LOOP

def train(model, dataloader, epochs, betas = betas, lr = 1e-3, save_every = 10):
    """
    Training loop for conditional diffusion.
    Dataloader yields (trajectory, observation, utilities_grid) tuples.
    """
    optimizer = Adam(model.parameters(), lr=lr)
    model.train()
    loss_vals = []
    stepcount = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for step, batch in tqdm(enumerate(dataloader)):
            stepcount.append(epoch * len(dataloader) + step)
            
            # Unpack batch: (trajectory, observation, utilities)
            traj, obs, util_grid = batch
            traj = traj.to(next(model.parameters()).device)
            obs = obs.to(next(model.parameters()).device)
            util_grid = util_grid.to(next(model.parameters()).device)
            
            batch_size = BATCH_SIZE
            t = torch.randint(0, T, (batch_size,), device=traj.device).long()
            
            # Compute loss with all conditions: trajectory, time, observation, utilities
            loss = get_loss(model, traj, t, util_grid)
            loss_vals.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}", flush = True)

        if (epoch + 1) % save_every == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
            }
            torch.save(checkpoint, f"checkpoints/checkpoint_epoch_{epoch+1}_conditional.pth")
            print(f"Checkpoint saved: checkpoints/checkpoint_epoch_{epoch+1}_conditional.pth")
        
    return loss_vals, stepcount

    
    
    
if __name__ == "__main__":
    

    os.makedirs("checkpoints", exist_ok=True)
    print("Checkpoint directory made", flush = True) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush = True)

    print("building Conditional UNET", flush = True)
    model = ConditionalUNET().to(device)  # NEW: specify observation dimension
    

    print("loading data", flush = True)
    data = TrajectoryDataset(trajectories, conditions, utilities)  # Include conditions and utilities
    # data = Subset(data, range(1))  # Use only the first sample for testing
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # # Use light noise schedule for single-sample debugging
    # betas_debug = linear_beta_schedule_light(timesteps=T)
    # alphas_debug = 1. - betas_debug
    # alphas_cumprod_debug = torch.cumprod(alphas_debug, axis=0)
    # alphas_cumprod_prev_debug = F.pad(alphas_cumprod_debug[:-1], (1, 0), value=1.0)
    # sqrt_recip_alphas_debug = torch.sqrt(1.0 / alphas_debug)
    # sqrt_alphas_cumprod_debug = torch.sqrt(alphas_cumprod_debug)
    # sqrt_one_minus_alphas_cumprod_debug = torch.sqrt(1. - alphas_cumprod_debug)
    
    # # Update global variables for debugging
    # import sys
    # globals()['betas'] = betas_debug
    # globals()['sqrt_alphas_cumprod'] = sqrt_alphas_cumprod_debug
    # globals()['sqrt_one_minus_alphas_cumprod'] = sqrt_one_minus_alphas_cumprod_debug
    
    # print("training conditional model with observations and utilities", flush = True)
    # print(f"Using DEBUG noise schedule (reduced noise for single-sample test)", flush=True)
    loss_vals, stepcount = train(model, dataloader, epochs=200, betas=betas, lr=5e-3, save_every=10)

    plt.plot(stepcount, loss_vals)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss (Conditional Diffusion)")
    plt.show()
    print("done training", flush = True)

    torch.save(model.state_dict(), "C:\\Users\\Aksha\\OneDrive\\Year 6\\Thesis\\scripts\\Diffusion\\checkpoints\\unet_diffusion_traj_conditional_final.pth")
    print("Final conditional model saved")
