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
- x_0: one sample is now shaped (1, 2, 17) [trajectory batch * coords * horizon]
- for now, since each trajectory is only 7 long, we keep the same time dim, we don't reduce time dim size. 
- data: ../trajectory_dataset.pt

- one thing I notice is that trajectory values are not neccesarily between 0 and 1, so we should normalize them to be between -1 and 1 (or 0 and 1) to make training easier.
'''




#first we define how much param beta change (how much information we change per step)
def linear_beta_schedule(timesteps, start = 0.0001, end = 0.02): 
    #linear schedule
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
BATCH_SIZE = 16


data_dict = torch.load(rf"C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts\trajectory_dataset.pt")
trajectories = data_dict["trajectories"].float()
conditions = data_dict["current_position"].float()

# Normalize trajectory coordinates to [0, 1] by dividing by 100.
# This preserves relative differences and makes noise scale meaningful for diffusion.
SCALE_FACTOR = 100.0
trajectories = trajectories / SCALE_FACTOR


class TrajectoryDataset(Dataset): 
    def __init__(self, trajectories, conditions = None): 
        self.trajectories = trajectories
        self.conditions = conditions

    def __len__(self): 
        return len(self.trajectories)
    
    def __getitem__(self, idx): 
        if self.conditions is None: 
            return self.trajectories[idx]
        
        else: 
            return self.trajectories[idx], self.conditions[idx]



data = TrajectoryDataset(trajectories)
# data_sub = Subset(data, range(10))  # Take only the first 100 samples for testing
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# sample = next(iter(dataloader)) 
# print("Sample trajectory shape:", sample.shape)  # Should be [BATCH_SIZE, 2, TRAJ_SIZE]

# Quick test of forward diffusion sampling for selected timesteps:
# for timestep in [1, 10, 25, 50, 90]:
#     t = torch.tensor([timestep], dtype=torch.long, device=sample.device)  # shape (1,)
#     x_noisy, noise = forward_diffusion_sample(sample, t)

#     print(f"t={timestep} | x_noisy shape={x_noisy.shape} | mse={F.mse_loss(x_noisy, sample).item():.6f}")

#     plt.figure()
#     plt.plot(x_noisy[0, 0].cpu(), x_noisy[0, 1].cpu(), marker='o', linestyle='-')
#     plt.title(f"Noisy Trajectory at timestep {timestep}")
#     plt.xlabel("coord 0")
#     plt.ylabel("coord 1")
#     plt.grid(True)
#     plt.show()




# quit() # The main training script will run through __main__, so these one-off tests are not needed here.

# The main training script will run through __main__, so these one-off tests are not needed here.
# If you want to visualize forward noise schedules, run a separate function or notebook.


#U-NET
#input: noisy image, timestep t
#neural networks share params across time so timesteps need to be embedded
#output: predicted noise


class Block(nn.Module): 

    def __init__(self, in_channels, out_channels, time_emb_dim, up = False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        if up:
            # upsample in the decoder path
            self.conv1 = nn.ConvTranspose1d(2*in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.transform = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        else:
            # downsample in the encoder path
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.transform = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        self.conv2 = nn.Conv1d(out_channels, out_channels,  kernel_size= 3, padding= 1) 
        # self.bnorm1 = nn.BatchNorm2d(out_channels)
        # self.bnorm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        #first convolution 
        h = self.relu(self.conv1(x))
        #time embedding 
        time_emb = self.relu(self.time_mlp(t)) #t in input shape (batch_size, time_emb_dim) 
        time_emb = time_emb[:, :, None] #reshape to (batch_size, time_emb_dim, 1)
        h = h + time_emb #add time embedding to feature maps (broadcasting)
        h = self.relu(self.conv2(h))

        return self.transform(h) #up or downsample depending on block type
        




##better to implement and understand UNET first 
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
        # TODO: Double check the ordering here
        return embeddings

class UNET(nn.Module):

    def __init__(self): 
        super().__init__()
        timestep_channels = 2 #(x, y)
        # down_channels = (32, 64, 128) #while maintaining 2*7 traj
        down_channels = (32, 64, 128)
        up_channels = down_channels[::-1]
        out_dim = 2
        time_emb_dim = 32 

        #TIME EMBEDDING - we embed the time step t into a vector of size time_emb_dim to be used in the UNET blocks

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )   

        self.conv0 = nn.Conv1d(timestep_channels, down_channels[0], kernel_size= 3, padding=1)

        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], time_emb_dim) for i in range(len(down_channels)-1)])

        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], time_emb_dim, up = True) for i in range(len(up_channels)-1)]) 

        self.output = nn.Conv1d(up_channels[-1], out_dim, kernel_size= 1) #final output is predicted noise with same shape as input traj


    def forward(self, x, t):

        #embed time 
        t = self.time_mlp(t)
        x = self.conv0(x)
        #downsampling

        residual_inputs = []
        for i, down in enumerate(self.downs):
            x = down(x, t)
            residual_inputs.append(x)

        for up in self.ups:
            skip = residual_inputs.pop()
            # align time dimension if needed due to rounding in down/up sampling
            if x.shape[2] != skip.shape[2]:
                if x.shape[2] < skip.shape[2]:
                    x = F.pad(x, (0, skip.shape[2] - x.shape[2]))
                else:
                    x = x[:, :, :skip.shape[2]]
            x = torch.cat((x, skip), dim=1)
            x = up(x, t)
        # residual_inputs = []
        # for down in self.downs:
        #     x = down(x, t)
        #     residual_inputs.append(x) #store inputs for skip connections
        # for up in self.ups:
        #     x = torch.cat((x, residual_inputs.pop()), dim=1) #skip connection
        #     x = up(x, t)
            
        return self.output(x)
        




#LOSS FUNCTION

def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t)
    noise_pred = model(x_noisy, t)
    return F.mse_loss(noise, noise_pred)



#sampling: one denoising step at a time
@torch.no_grad()
def sample_timestep(x, t): 
    """
    Calls the model, predicts the noise in the traj at timestep t, returns the denoised traj (model predicts the noise)
    
    """


    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t[0].item() == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise #we must add noise again (gives many possible images instead of mode collapse)
    


@torch.no_grad()
def sample_plot_traj(): 
    model.eval()
    traj_size = TRAJ_SIZE
    traj = torch.randn((1, 2, traj_size)).to(next(model.parameters()).device) #start with pure noise
    plt.figure(figsize=(15, 15))
    plt.axis("off")
    num_images = 10
    stepsize = int(T / num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, dtype=torch.long, device = device)
        traj = sample_timestep(traj, t) #here we denoise for T timesteps
        # Edit: This is to maintain the natural range of the distribution
        traj = torch.clamp(traj, -1.0, 1.0)
        if i % stepsize == 0:
            print(traj)
    # plt.savefig("attempt.png")             




#TRAINING LOOP

def train(model, dataloader, epochs, betas = betas, lr = 1e-3, save_every = 10):
    optimizer = Adam(model.parameters(), lr=lr)
    model.train()
    loss_vals = []
    stepcount = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for step, traj in tqdm(enumerate(dataloader)): #dataloader has images + labels remember
            # if step >= 10: 
            #     continue #just train on 10 batches per epoch for speed, remove this later
            stepcount.append(epoch * len(dataloader) + step)
            # print(len(dataloader))
            traj = traj.to(next(model.parameters()).device)
            batch_size = BATCH_SIZE
            t = torch.randint(0, T, (batch_size,), device=traj.device).long() #randomly sample a timestep for each image in the batch
            loss = get_loss(model, traj, t)
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
            torch.save(checkpoint, f"checkpoints/checkpoint_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved: checkpoints/checkpoint_epoch_{epoch+1}.pth")
        
    return loss_vals, stepcount

    
    
    
if __name__ == "__main__":
    

    os.makedirs("checkpoints", exist_ok=True)
    print("Checkpoint directory made", flush = True) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush = True)

    print("building UNET", flush = True)
    model = UNET().to(device)
    

    # summary(
    # model,
    # input_data=(
    #     torch.randn(1, 2, 7).to(device),                     # x: [B, C, L]
    #     torch.randint(0, T, (1,), dtype=torch.long).to(device)  # t
    # ),
    # col_names=["input_size", "output_size", "num_params"],
    # )

    # quit()
    print("loading data", flush = True)
    data = TrajectoryDataset(trajectories)
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print("training model", flush = True)
    loss_vals, stepcount = train(model, dataloader, epochs=120, betas=betas, lr=1e-3, save_every=10)

    plt.plot(stepcount, loss_vals)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()
    print("done training", flush = True)

    torch.save(model.state_dict(), "checkpoints/unet_diffusion_traj_final.pth")
    print("Final model saved to checkpoints/unet_diffusion_traj_final.pth")

    # sample_plot_image()



