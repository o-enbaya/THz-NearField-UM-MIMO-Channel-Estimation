import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

"""
Offline Meta-Training of the DCS Generator (MAML + RIP Loss).
Thesis Reference: Algorithm 1, Table 4.2, Chapter 3 (Section 3.7 - 3.8).

Outer Loop: Adam (lr=0.0002, betas=(0.9, 0.99), batch_size=50, 300 epochs)
Inner Loop: T=20 gradient steps on latent code z
Auxiliary RIP Loss: Enforces distance preservation on sparse channels
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from python_dcs.models.generator import DCSChannelGenerator
from python_dcs.models.measurement_operator import THzMeasurementOperator
from python_dcs.dataset.thz_dataset import THzChannelDataset

def rip_auxiliary_loss(generator, latent_dim, batch_size, device='cpu'):
    """
    Computes auxiliary Restricted Isometry Property (RIP) loss L_F
    over pairs of sampled latent vectors (Eq 3.12 / Algorithm 1).
    """
    z1 = torch.randn(batch_size, latent_dim, device=device)
    z2 = torch.randn(batch_size, latent_dim, device=device)
    z1 = z1 / torch.norm(z1, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    z2 = z2 / torch.norm(z2, p=2, dim=-1, keepdim=True).clamp(min=1e-8)

    H1 = generator(z1)
    H2 = generator(z2)

    dist_z = torch.norm(z1 - z2, p=2, dim=-1)
    dist_H = torch.norm((H1 - H2).view(batch_size, -1), p=2, dim=-1)

    # Distance preservation penalty
    loss_rip = torch.mean((dist_H - dist_z) ** 2)
    return loss_rip

def train_dcs_maml(dataset, num_epochs=300, batch_size=50, inner_steps=20,
                   lr_inner=0.1, lr_outer=0.0002, lambda_rip=0.05,
                   device='cpu', save_path='dcs_generator_trained.pth'):
    """
    Implements Algorithm 1: Offline Meta-Training of DCS-MAML.
    """
    generator = DCSChannelGenerator(latent_dim=128, out_channels=16, nr=32, nt=256).to(device)
    meas_op = THzMeasurementOperator(nr=32, nt=256, k_sub=8, m_pilots=100, device=device)

    # Table 4.2: Outer optimizer Adam (lr=0.0002, betas=(0.9, 0.99))
    outer_optimizer = optim.Adam(generator.parameters(), lr=lr_outer, betas=(0.9, 0.99))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"Starting DCS-MAML Meta-Training for {num_epochs} epochs (Batch size={batch_size})...")

    for epoch in range(1, num_epochs + 1):
        generator.train()
        epoch_loss = 0.0

        for batch_idx, H_true in enumerate(dataloader):
            H_true = H_true.to(device)
            B = H_true.size(0)

            # 1. Simulate received pilots y_i = Phi * H_true + noise
            with torch.no_grad():
                y_pilot = meas_op(H_true)
                y_pilot = meas_op.add_noise(y_pilot, snr_db=10.0)

            # 2. Task-specific Inner Loop: Optimize latent z_i
            z_i = torch.randn(B, 128, device=device, requires_grad=True)
            inner_optimizer = optim.Adam([z_i], lr=lr_inner, betas=(0.9, 0.99))

            for j in range(inner_steps):
                inner_optimizer.zero_grad()
                H_cand = generator(z_i)
                y_cand = meas_op(H_cand)
                inner_loss = torch.mean((y_pilot - y_cand) ** 2) + 0.01 * torch.mean(z_i ** 2)
                inner_loss.backward()
                inner_optimizer.step()

                with torch.no_grad():
                    z_i.div_(torch.norm(z_i, p=2, dim=-1, keepdim=True).clamp(min=1e-8))

            # 3. Outer Meta-Update: Evaluate on ground truth H_true
            outer_optimizer.zero_grad()
            H_recon = generator(z_i)

            # Normalized Mean Squared Error (Eq 4.1 / Algorithm 1 step 14)
            num = torch.sum((H_true - H_recon) ** 2, dim=(1, 2, 3))
            den = torch.sum(H_true ** 2, dim=(1, 2, 3)).clamp(min=1e-8)
            loss_g = torch.mean(num / den)

            # Auxiliary RIP Loss (step 15)
            loss_rip = rip_auxiliary_loss(generator, 128, B, device=device)

            total_loss = loss_g + lambda_rip * loss_rip
            total_loss.backward()
            outer_optimizer.step()

            epoch_loss += total_loss.item()

        if epoch % 25 == 0 or epoch == 1:
            avg_loss = epoch_loss / max(1, len(dataloader))
            print(f"Epoch [{epoch}/{num_epochs}] - Meta-Loss: {avg_loss:.4f} (NMSE: {loss_g.item():.4f})")

    torch.save(generator.state_dict(), save_path)
    print(f"Model saved successfully to {save_path}")
    return generator

if __name__ == '__main__':
    dataset = THzChannelDataset()
    train_dcs_maml(dataset, num_epochs=2, batch_size=10, inner_steps=5)
