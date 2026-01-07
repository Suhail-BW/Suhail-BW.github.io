---
layout: post
title: "Diffusion Models: The Mathematics Behind Image and Video Generation"
date: 2024-01-22
description: "A comprehensive guide to understanding diffusion models, from foundational theory to state-of-the-art applications in image and video generation."
---

# Introduction

Diffusion models have revolutionized generative AI, powering tools like DALL-E 2, Stable Diffusion, and Midjourney. But what makes them so effective? In this post, I'll break down the mathematics, intuition, and implementation details behind diffusion models for both image and video generation.

## What Are Diffusion Models?

Diffusion models are generative models inspired by non-equilibrium thermodynamics. They learn to generate data by reversing a gradual noising process. The key insight: if we can learn to remove noise step-by-step, we can generate realistic samples from pure noise.

### The Core Idea

The process consists of two phases:

1. **Forward Diffusion Process**: Gradually add Gaussian noise to data until it becomes pure noise
2. **Reverse Diffusion Process**: Learn to denoise, starting from noise and recovering the original data

This is mathematically elegant and surprisingly effective!

## Mathematical Foundation

### Forward Process

The forward process is a fixed Markov chain that gradually adds Gaussian noise to the data:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

Where:
- $x_0$ is the original data
- $x_T$ is pure Gaussian noise
- $\beta_t$ is the noise schedule

A beautiful property: we can sample $x_t$ at any timestep directly:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$$

Where $\bar{\alpha}_t = \prod_{i=1}^{t}(1-\beta_i)$ and $\epsilon \sim \mathcal{N}(0, I)$

### Reverse Process

The reverse process learns to denoise:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

The neural network $\mu_\theta$ predicts the mean of the denoising distribution.

### Training Objective

The variational lower bound leads to a simplified training objective:

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ || \epsilon - \epsilon_\theta(x_t, t) ||^2 \right]$$

We simply train the network to predict the noise that was added!

## DDPM: Denoising Diffusion Probabilistic Models

Let's implement a basic DDPM architecture:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionModel(nn.Module):
    def __init__(self, image_size=64, in_channels=3, model_channels=128,
                 num_res_blocks=2, attention_resolutions=(8, 16)):
        super().__init__()
        self.image_size = image_size

        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # U-Net architecture
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Downsampling
        self.down_blocks = nn.ModuleList([
            ResBlock(model_channels, model_channels * 2, time_embed_dim),
            ResBlock(model_channels * 2, model_channels * 4, time_embed_dim),
        ])

        # Middle
        self.middle_block = ResBlock(model_channels * 4, model_channels * 4, time_embed_dim)

        # Upsampling
        self.up_blocks = nn.ModuleList([
            ResBlock(model_channels * 4, model_channels * 2, time_embed_dim),
            ResBlock(model_channels * 2, model_channels, time_embed_dim),
        ])

        # Output
        self.output_conv = nn.Conv2d(model_channels, in_channels, 3, padding=1)

    def forward(self, x, timesteps):
        # Time embedding
        t_emb = self.get_timestep_embedding(timesteps)
        t_emb = self.time_embed(t_emb)

        # U-Net forward pass
        h = self.input_conv(x)

        # Downsampling with skip connections
        skip_connections = []
        for block in self.down_blocks:
            h = block(h, t_emb)
            skip_connections.append(h)
            h = F.avg_pool2d(h, 2)

        # Middle
        h = self.middle_block(h, t_emb)

        # Upsampling with skip connections
        for block in self.up_blocks:
            h = F.interpolate(h, scale_factor=2, mode='nearest')
            h = torch.cat([h, skip_connections.pop()], dim=1)
            h = block(h, t_emb)

        # Output
        return self.output_conv(h)

    def get_timestep_embedding(self, timesteps, dim=128):
        """
        Create sinusoidal timestep embeddings.
        """
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResBlock(nn.Module):
    """Residual block with time conditioning."""
    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_embed_dim, out_channels)
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time conditioning
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.shortcut(x)
```

## Training Diffusion Models

Here's the training loop:

```python
def train_diffusion_model(model, dataloader, num_epochs=100,
                          timesteps=1000, device='cuda'):
    """
    Train a diffusion model using the simplified objective.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Define noise schedule (linear)
    betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            batch_size = images.shape[0]

            # Sample random timesteps
            t = torch.randint(0, timesteps, (batch_size,), device=device)

            # Sample noise
            noise = torch.randn_like(images)

            # Get noisy images at timestep t
            alpha_t = alphas_cumprod[t][:, None, None, None]
            noisy_images = torch.sqrt(alpha_t) * images + \
                          torch.sqrt(1 - alpha_t) * noise

            # Predict noise
            predicted_noise = model(noisy_images, t)

            # Compute loss
            loss = F.mse_loss(predicted_noise, noise)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    return model
```

## Sampling (Generation)

The generation process reverses the diffusion:

```python
@torch.no_grad()
def sample_ddpm(model, image_size, batch_size=16, timesteps=1000, device='cuda'):
    """
    Generate images using DDPM sampling.
    """
    model.eval()

    # Start from pure noise
    x = torch.randn(batch_size, 3, image_size, image_size).to(device)

    # Define noise schedule
    betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # Reverse process
    for t in reversed(range(timesteps)):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Predict noise
        predicted_noise = model(x, t_batch)

        # Compute denoising parameters
        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        beta_t = betas[t]

        # Denoising step
        x = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
        )

        # Add noise (except at last step)
        if t > 0:
            noise = torch.randn_like(x)
            x = x + torch.sqrt(beta_t) * noise

    return x
```

## DDIM: Faster Sampling

Denoising Diffusion Implicit Models (DDIM)[^1] enable faster sampling by using a deterministic process:

```python
@torch.no_grad()
def sample_ddim(model, image_size, batch_size=16, timesteps=1000,
                ddim_steps=50, device='cuda'):
    """
    Generate images using DDIM (faster sampling).
    """
    model.eval()

    # Use subset of timesteps
    skip = timesteps // ddim_steps
    timestep_sequence = range(0, timesteps, skip)

    # Start from noise
    x = torch.randn(batch_size, 3, image_size, image_size).to(device)

    # Define schedule
    betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alphas_cumprod = torch.cumprod(1 - betas, dim=0)

    for i in reversed(range(len(timestep_sequence))):
        t = timestep_sequence[i]
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Predict noise
        predicted_noise = model(x, t_batch)

        # Get alpha values
        alpha_cumprod_t = alphas_cumprod[t]
        alpha_cumprod_t_prev = alphas_cumprod[timestep_sequence[i-1]] if i > 0 else 1.0

        # Predict x0
        pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / \
                  torch.sqrt(alpha_cumprod_t)

        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev) * predicted_noise

        # Deterministic update
        x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt

    return x
```

## Latent Diffusion Models (Stable Diffusion)

Stable Diffusion operates in latent space, making it much more efficient:

```python
class LatentDiffusionModel(nn.Module):
    """
    Simplified Latent Diffusion Model architecture.
    """
    def __init__(self, vae, unet, text_encoder):
        super().__init__()
        self.vae = vae  # Variational Autoencoder
        self.unet = unet  # Denoising U-Net
        self.text_encoder = text_encoder  # CLIP text encoder

    def encode_images(self, images):
        """Encode images to latent space."""
        return self.vae.encode(images).latent_dist.sample() * 0.18215

    def decode_latents(self, latents):
        """Decode latents to images."""
        return self.vae.decode(latents / 0.18215).sample

    @torch.no_grad()
    def generate(self, prompt, num_inference_steps=50, guidance_scale=7.5):
        """
        Generate image from text prompt.

        Args:
            prompt: Text description
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance strength
        """
        # Encode text
        text_embeddings = self.text_encoder(prompt)

        # Also get unconditional embeddings for classifier-free guidance
        uncond_embeddings = self.text_encoder("")

        # Start from random latent
        latents = torch.randn(1, 4, 64, 64).to(self.device)

        # Denoising loop
        for t in self.scheduler.timesteps:
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)

            # Predict noise
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=torch.cat([uncond_embeddings, text_embeddings])
            )

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * \
                        (noise_pred_text - noise_pred_uncond)

            # Denoise step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode to image
        image = self.decode_latents(latents)

        return image
```

## Video Diffusion Models

Extending to video requires modeling temporal consistency:

```python
class VideoDiffusionModel(nn.Module):
    """
    3D U-Net for video generation.
    """
    def __init__(self, num_frames=16, image_size=64, channels=3):
        super().__init__()
        self.num_frames = num_frames

        # 3D convolutions for temporal modeling
        self.input_conv = nn.Conv3d(channels, 64, kernel_size=3, padding=1)

        # Spatial-temporal blocks
        self.down_blocks = nn.ModuleList([
            SpatialTemporalBlock(64, 128),
            SpatialTemporalBlock(128, 256),
            SpatialTemporalBlock(256, 512),
        ])

        self.middle_block = SpatialTemporalBlock(512, 512)

        self.up_blocks = nn.ModuleList([
            SpatialTemporalBlock(512, 256),
            SpatialTemporalBlock(256, 128),
            SpatialTemporalBlock(128, 64),
        ])

        self.output_conv = nn.Conv3d(64, channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        """
        Args:
            x: Video tensor of shape (B, C, T, H, W)
            t: Timestep
        """
        h = self.input_conv(x)

        # Downsampling
        skip_connections = []
        for block in self.down_blocks:
            h = block(h, t)
            skip_connections.append(h)
            h = F.avg_pool3d(h, kernel_size=(1, 2, 2))

        # Middle
        h = self.middle_block(h, t)

        # Upsampling
        for block in self.up_blocks:
            h = F.interpolate(h, scale_factor=(1, 2, 2), mode='trilinear')
            h = torch.cat([h, skip_connections.pop()], dim=1)
            h = block(h, t)

        return self.output_conv(h)


class SpatialTemporalBlock(nn.Module):
    """
    Block that processes both spatial and temporal dimensions.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Spatial convolution
        self.spatial_conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )
        # Temporal convolution
        self.temporal_conv = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=(3, 1, 1), padding=(1, 0, 0)
        )
        self.norm = nn.GroupNorm(32, out_channels)

    def forward(self, x, t):
        h = self.spatial_conv(x)
        h = self.temporal_conv(h)
        h = self.norm(h)
        return F.silu(h)
```

## Key Innovations

### 1. Classifier-Free Guidance

Improves sample quality by guiding generation without a separate classifier:

$$\tilde{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + s \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$

Where:
- $c$ is the conditioning (e.g., text prompt)
- $s$ is the guidance scale
- $\emptyset$ represents unconditional generation

### 2. Cascaded Diffusion

Generate low-resolution images first, then upscale with super-resolution diffusion models. Used in DALL-E 2 and Imagen.

### 3. Latent Space Diffusion

Operating in compressed latent space (Stable Diffusion) reduces computational cost by ~10x while maintaining quality.

## Practical Applications

### Image Generation
- **Text-to-Image**: Stable Diffusion, DALL-E 2, Midjourney
- **Image Editing**: Inpainting, outpainting, style transfer
- **Super-Resolution**: Enhancing image quality

### Video Generation
- **Text-to-Video**: Generate videos from descriptions
- **Frame Interpolation**: Create smooth transitions
- **Video Editing**: Modify existing videos

## Advantages and Limitations

**Advantages:**
- High-quality samples
- Stable training (compared to GANs)
- Flexible conditioning
- Principled mathematical framework

**Limitations:**
- Slow sampling (requires many steps)
- High computational cost
- Difficulty with fine details
- Mode coverage issues

## Future Directions

Current research focuses on:
- **Faster sampling**: Flow matching, consistency models
- **Better control**: Spatial control, attribute manipulation
- **3D generation**: NeRF-based diffusion models
- **Longer videos**: Extending temporal coherence
- **Efficient architectures**: Reducing memory and computation

## Conclusion

Diffusion models represent a paradigm shift in generative modeling. Their mathematical elegance, combined with practical effectiveness, has made them the go-to approach for image and video generation. As research continues, we can expect even more impressive applications.

The key takeaway: by learning to reverse a simple noise-adding process, we can generate remarkably complex and realistic content. This simple idea has profound implications for creative AI.

---

## References

[^1]: Song, J., Meng, C., & Ermon, S. (2020). "Denoising Diffusion Implicit Models." *arXiv:2010.02502*. [Link](https://arxiv.org/abs/2010.02502)

[^2]: Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS*. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

[^3]: Rombach, R., et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR*. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)

[^4]: Ho, J., et al. (2022). "Video Diffusion Models." *NeurIPS*. [arXiv:2204.03458](https://arxiv.org/abs/2204.03458)

[^5]: Ramesh, A., et al. (2022). "Hierarchical Text-Conditional Image Generation with CLIP Latents." *arXiv:2204.06125*. [Link](https://arxiv.org/abs/2204.06125)
