import matplotlib.pyplot as plt
from tensorboard import summary
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_dataset
from PIL import Image
import numpy as np
import math
from torch import nn
from torch.optim import Adam
import os
from torchinfo import summary
from torchviz import make_dot


os.makedirs("checkpoints", exist_ok=True)

def show_images(dataset, num_samples=4, cols=4):
    plt.figure(figsize=(15, 15))
    
    for i in range(num_samples):
        img, label = dataset[i]   # dataset[i] = (image, label)

        # Convert from (C, H, W) to (H, W, C)
        img = img.permute(1, 2, 0)

        plt.subplot(num_samples // cols + 1, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(str(label))

    plt.tight_layout()
    plt.show()

transform = transforms.Compose([
    transforms.ToTensor(),
])

# dataset = torchvision.datasets.CIFAR10(
#     root="./data",
#     train=True,
#     download=True,
#     transform=transform
# )

# show_images(dataset)

#first we define how much param beta change (how much information we change per step)
def linear_beta_schedule(timesteps, start = 0.0001, end = 0.02): 
    #linear schedule
    return torch.linspace(start, end, timesteps)
    

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, betas): 
    noise = torch.randn_like(x_0)

    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)


    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise



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
IMG_SIZE = 64
BATCH_SIZE = 128

def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)  # Scale to [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train_path = "./stanford_cars_local/train"
    test_path = "./stanford_cars_local/test"

    train_dataset = torchvision.datasets.ImageFolder(
        root=train_path,
        transform=data_transform
    )

    test_dataset = torchvision.datasets.ImageFolder(
        root=test_path,
        transform=data_transform
    )

    return ConcatDataset([train_dataset, test_dataset])


def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))

data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)



## forward process 

# image = next(iter(dataloader))[0] # Get a batch of images

# plt.figure(figsize=(15, 15))
# plt.axis("off")
# num_images = 5 

# stepsize = int(T / num_images) #forward process stepsize

# for idx in range(0, T, stepsize): 
#     t = torch.tensor([idx]).type(torch.int64)
#     img, noise = forward_diffusion_sample(image, t, betas)
#     plt.subplot(1, num_images, idx // stepsize + 1)
#     show_tensor_image(img)
# plt.show()


#U-NET
#input: noisy image, timestep t
#neural networks share params across time so timesteps need to be embedded
#output: predicted noise


class Block(nn.Module): 

    def __init__(self, in_channels, out_channels, time_emb_dim, up = False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels) #don't need activation function? 

        if up: 
            self.conv1 = nn.Conv2d(2*in_channels, out_channels, kernel_size = 3, padding = 1) #making out_ch feature maps
            self.transform = nn.ConvTranspose2d(out_channels, out_channels, kernel_size = 4, stride = 2, padding = 1) #increasing dim
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1) #making out_ch feature maps
            self.transform = nn.Conv2d(out_channels, out_channels, kernel_size = 4, stride = 2, padding = 1) #decreasing dim

        self.conv2 = nn.Conv2d(out_channels, out_channels,  kernel_size= 3, padding= 1)
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.bnorm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        #first convolution 
        h = self.bnorm1(self.relu(self.conv1(x)))
        #time embedding 
        time_emb = self.relu(self.time_mlp(t)) #t in input shape (batch_size, time_emb_dim) 
        time_emb = time_emb[(..., ) + (None, ) * 2] #reshape to (batch_size, time_emb_dim, 1, 1) for broadcasting 
        h = h + time_emb #add time embedding to feature maps (broadcasting)
        h = self.bnorm2(self.relu(self.conv2(h)))

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
        image_channels = 3 
        down_channels = (64, 128, 256, 512, 1024) #turn 64x64 image into 2x2 image after 5 downsamplings (64 -> 32 -> 16 -> 8 -> 4 -> 2)
        up_channels = down_channels[::-1]
        out_dim = 3
        time_emb_dim = 32

        #TIME EMBEDDING - we embed the time step t into a vector of size time_emb_dim to be used in the UNET blocks

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )   

        self.conv0 = nn.Conv2d(image_channels, down_channels[0], kernel_size= 3, padding=1)

        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], time_emb_dim) for i in range(len(down_channels)-1)])

        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], time_emb_dim, up = True) for i in range(len(up_channels)-1)]) 

        self.output = nn.Conv2d(up_channels[-1], out_dim, kernel_size= 1) #final output is predicted noise with same shape as input image


    def forward(self, x, t):

        #embed time 
        t = self.time_mlp(t)
        x = self.conv0(x)
        #downsampling
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x) #store inputs for skip connections
        for up in self.ups:
            x = torch.cat((x, residual_inputs.pop()), dim=1) #skip connection
            x = up(x, t)
            
        return self.output(x)
        


model = UNET()
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))


#LOSS FUNCTION

def get_loss(model, x_0, t, betas):
    x_noisy, noise = forward_diffusion_sample(x_0, t, betas)
    noise_pred = model(x_noisy, t)
    return F.mse_loss(noise, noise_pred)



#sampling: one denoising step at a time
@torch.no_grad()
def sample_timestep(x, t): 
    """
    Calls the model, predicts the noise in the image at timestep t, returns the denoised image (model predicts the noise)
    
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
def sample_plot_image(): 
    model.eval()
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size)).to(next(model.parameters()).device) #start with pure noise
    plt.figure(figsize=(15, 15))
    plt.axis("off")
    num_images = 10
    stepsize = int(T / num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, dtype=torch.long, device = device)
        img = sample_timestep(img, t) #here we denoise for T timesteps
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu())
    plt.savefig("attempt.png")             




#TRAINING LOOP

def train(model, dataloader, epochs = 10, betas = betas, lr = 1e-3, save_every = 10):
    optimizer = Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for step, (images, _) in enumerate(dataloader): #dataloader has images + labels remember
            images = images.to(next(model.parameters()).device)
            batch_size = images.shape[0]
            t = torch.randint(0, T, (batch_size,), device=images.device).long() #randomly sample a timestep for each image in the batch
            loss = get_loss(model, images, t, betas)

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
            torch.save(checkpoint, f"checkpoint_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved: checkpoint_epoch_{epoch+1}.pth")

    
    
    
if __name__ == "__main__":
    
    summary(
    model,
    input_data=(
        torch.randn(1, 3, 64, 64),          # x
        torch.randint(0, T, (1,), dtype=torch.long)  # t
    ),
    col_names=["input_size", "output_size", "num_params"],
    )
    
    os.makedirs("checkpoints", exist_ok=True)
    print("Checkpoint directory made", flush = True) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush = True)

    print("building UNET", flush = True)
    model = UNET().to(device)
    
    print("loading data", flush = True)
    data = load_transformed_dataset()
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print("training model", flush = True)
    train(model, dataloader, epochs=1, betas=betas, lr=1e-3, save_every=1)
    
    print("done training", flush = True)

    torch.save(model.state_dict(), "checkpoints/unet_diffusion_cifar10_final.pth")
    print("Final model saved to checkpoints/unet_diffusion_cifar10_final.pth")

    sample_plot_image()



