import os
import torch
import matplotlib.pyplot as plt

from base_1d_conditional import (
    ConditionalUNET,
    T,
    TRAJ_SIZE,
    get_index_from_list,
    betas,
    sqrt_one_minus_alphas_cumprod,
    sqrt_recip_alphas,
    posterior_variance,
    trajectories,
    conditions,
    utilities
)


def sample_timestep_model(model, x, t, conditions, utilities):
    # t: 1D tensor of shape (batch_size,)
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t, utilities) / sqrt_one_minus_alphas_cumprod_t)
    model_mean[:, :, 0] = conditions  # Enforce initial condition at the first position
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t[0].item() == 0:
        return model_mean

    noise = torch.randn_like(x)
    return model_mean + torch.sqrt(posterior_variance_t) * noise


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Build model and load checkpoint
    model = ConditionalUNET().to(device)
    ckpt_path = 'checkpoints/checkpoint_epoch_120.pth'
    if not os.path.exists(ckpt_path):
        # fallback to end model if explicit checkpoint not available
        ckpt_path = 'C:\\Users\\Aksha\\OneDrive\\Year 6\\Thesis\\scripts\\Diffusion\\checkpoints\\unet_diffusion_traj_conditional_final.pth'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: checkpoints/checkpoint_epoch_120.pth or unet_diffusion_traj_final.pth')

    ckpt = torch.load(ckpt_path, map_location=device)
    print(f'Loaded checkpoint: {ckpt_path}')
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Choose a test trajectory (or random noise if you prefer)
    if trajectories is not None and len(trajectories) > 0:
        x0 = trajectories[:1].to(device)  # [1, 2, TRAJ_SIZE]
        c0 = conditions[:1].to(device)  # [1, 2]
        u0 = utilities[:1].to(device)  # [1, 1, 51, 51]
        print('Using sample from dataset; min/max:', x0.min().item(), x0.max().item())
    else:
        x0 = torch.randn((1, 2, TRAJ_SIZE), device=device)
        print('Using random x0 input')



    # Start from pure noise for sampling
    x = torch.randn_like(x0).to(device)

    # Track interpolated paths for plotting every few steps
    frames = {}
    snapshot_steps = sorted({0, 5, 10, 25, 50, 75, 99})

    for tstep in reversed(range(T)):
        cur_t = torch.tensor([tstep], dtype=torch.long, device=device)
        x = sample_timestep_model(model, x, cur_t, c0, u0)

        if tstep in snapshot_steps:
            frames[tstep] = x.detach().cpu()[0].clone()

    # Plot final denoised trajectory and some snapshots (denormalize back to [0,100])
    scale_factor = 100.0
    plt.figure(figsize=(12, 8))
    for i, step in enumerate(snapshot_steps):
        traj = frames.get(step)
        if traj is None:
            continue
        traj = traj * scale_factor
        plt.subplot(2, 4, i + 1)
        plt.plot(traj[0].numpy(), traj[1].numpy(), marker='o', linestyle='-')
        plt.title(f't={step} (scaled)')
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.grid(True)

    plt.suptitle('Denoised trajectory snapshots during reverse diffusion (0-100 range)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Optional: plot final trajectory alone
    final_traj = x.detach().cpu()[0] * scale_factor
    plt.figure(figsize=(6, 6))
    plt.plot(final_traj[0].numpy(), final_traj[1].numpy(), marker='o', linestyle='-', label = "Final Denoised")
    plt.plot(x0[0, 0].cpu().numpy() * scale_factor, x0[0, 1].cpu().numpy() * scale_factor, marker='x', linestyle='--', label='Original')
    plt.scatter(c0[0, 0].cpu().numpy() * scale_factor, c0[0, 1].cpu().numpy() * scale_factor, color='red', marker='*', s=200, label='Initial condition')
    plt.title('Final denoised trajectory (0-100 range)')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.ylabel('Y Position (m)')
    plt.xlabel('X Position (m)')
    plt.legend()
    plt.grid(True)
    plt.show()


