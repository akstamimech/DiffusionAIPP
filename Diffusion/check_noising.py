import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from base_1d import (trajectories, forward_diffusion_sample, T, get_index_from_list,
                      sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                      UNET, betas, sqrt_recip_alphas, posterior_variance)


def evaluate_noising(batch, timesteps):
    results = []
    for t_val in timesteps:
        t = torch.full((batch.shape[0],), t_val, dtype=torch.long, device=batch.device)
        x_noisy, noise = forward_diffusion_sample(batch, t)

        mse_to_orig = F.mse_loss(x_noisy, batch).item()
        noise_std = noise.std().item()
        x_noisy_std = x_noisy.std().item()
        results.append((t_val, mse_to_orig, noise_std, x_noisy_std))

    return results


def sample_timestep_model(model, x, t):
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t.item() == 0:
        return model_mean
    noise = torch.randn_like(x)
    return model_mean + torch.sqrt(posterior_variance_t) * noise


def plot_denoising_trajectory(model, x0, steps=None, device="cpu"):
    # Choose first trajectory in batch for plotting
    x0 = x0[0:1].to(device)
    if steps is None:
        steps = list(range(T-1, -1, -1))

    z = torch.randn_like(x0).to(device)
    x = z

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(18, 8))
    axes = axes.flatten()

    # Choose frames to plot evenly across steps
    selected = set(torch.linspace(T-1, 0, len(axes), dtype=torch.long).tolist())

    frame = 0
    for t in range(T-1, -1, -1):
        ts = torch.full((1,), t, dtype=torch.long, device=device)
        x = sample_timestep_model(model, x, ts)

        if t in selected:
            x_cpu = x[0].detach().cpu()
            x_plot = x_cpu.numpy()
            axes[frame].plot(x_plot[0], x_plot[1], marker='o', linestyle='-')
            axes[frame].set_title(f"t={t}")
            axes[frame].set_xlim(0, 1)
            axes[frame].set_ylim(0, 1)
            frame += 1

    fig.suptitle("Denoising trajectory sequence (sampled)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Use a small subset for quick checks
    test_batch = trajectories[:4].to(device)

    # Ensure data is in scaled range (expected in [0, 1])
    print("batch min/max", test_batch.min().item(), test_batch.max().item())

    timesteps = [0, 1, 5, 10, 25, 50, 75, 99]
    results = evaluate_noising(test_batch, timesteps)

    print("timestep | mse(x_t,x0) | noise_std | x_t_std")
    for t_val, mse, nstd, xtstd in results:
        print(f"{t_val:>3}      | {mse:.6f}     | {nstd:.6f} | {xtstd:.6f}")

    # Verify increasing variance behavior
    diffs = [results[i+1][1] - results[i][1] for i in range(len(results)-1)]
    print("mse increases between increments:", diffs)

    # Check get_index_from_list function consistency
    # Directly check that sqrt_alphas_cumprod pairwise is decreasing
    alpha_vals = get_index_from_list(sqrt_alphas_cumprod, torch.tensor(timesteps, dtype=torch.long), test_batch.shape)
    print("sqrt_alphas_cumprod extracted:\n", alpha_vals[:,0,0])
    alpha_vals2 = get_index_from_list(sqrt_one_minus_alphas_cumprod, torch.tensor(timesteps, dtype=torch.long), test_batch.shape)
    print("sqrt_one_minus_alphas_cumprod extracted:\n", alpha_vals2[:,0,0])

    # Build and (optionally) load model for reverse-denoising visualization
    model = UNET().to(device)
    checkpoint_path = "checkpoints/unet_diffusion_traj_final.pth"
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Checkpoint not found at {checkpoint_path}. Using randomly initialized model.")

    model.eval()
    plot_denoising_trajectory(model, test_batch, device=device)

