"""
Online Real-Time Channel Estimation (Inference Phase).
Thesis Reference: Algorithm 2, Chapter 3 (Section 3.8.1).

Optimizes latent vector z in R^{d_z} via Adam (lr=0.1) over T iterations
with unit sphere projection z <- z / ||z||_2, then outputs estimated
channel tensor H_hat = G_theta(z).
"""

import torch
import torch.optim as optim

def estimate_channel_online(generator, measurement_op, y_received, snr_db,
                            num_steps=20, lr=0.1, alpha_reg=0.01, device='cpu'):
    """
    Implements Algorithm 2: Online Latent Optimization.

    Args:
        generator (nn.Module): Pre-trained and frozen DCS generator G_theta.
        measurement_op (nn.Module): Measurement operator Phi.
        y_received (torch.Tensor): Observed compressed pilot signal (B, M_pilots).
        snr_db (float): Current channel SNR in dB.
        num_steps (int): Online optimization steps T (default 20).
        lr (float): Latent space Adam learning rate (Table 4.2: 0.1).
        alpha_reg (float): Regularization coefficient.
        device (str): Computation device ('cpu' or 'cuda').

    Returns:
        H_est (torch.Tensor): Estimated channel tensor (B, 2K, Nr, Nt).
        losses (list): Evolution of inner loss.
    """
    generator.eval()
    B = y_received.size(0)
    latent_dim = generator.latent_dim

    # 1. Initialize latent code z from standard normal
    z = torch.randn(B, latent_dim, device=device, requires_grad=True)

    # Table 4.2: Adam optimizer with lr=0.1, betas=(0.9, 0.99)
    optimizer = optim.Adam([z], lr=lr, betas=(0.9, 0.99))

    losses = []
    for step in range(num_steps):
        optimizer.zero_grad()

        # Generate candidate channel
        H_cand = generator(z)

        # Apply measurement operator Phi
        y_cand = measurement_op(H_cand)

        # Inner loss (Algorithm 2, Step 3)
        meas_loss = torch.mean((y_received - y_cand) ** 2)
        reg_loss = alpha_reg * torch.mean(z ** 2)
        total_loss = meas_loss + reg_loss

        total_loss.backward()
        optimizer.step()

        # Step 5: Project onto unit sphere
        with torch.no_grad():
            z.div_(torch.norm(z, p=2, dim=-1, keepdim=True).clamp(min=1e-8))

        losses.append(total_loss.item())

    # Final step: recover channel
    with torch.no_grad():
        H_est = generator(z)

    return H_est, losses
