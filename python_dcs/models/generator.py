"""
Continuous Generative Prior for THz UM-MIMO Channel Estimation.
Thesis Reference: Chapter 3 (Section 3.6, 3.8), Chapter 4 (Section 4.3).

Maps low-dimensional latent noise z in R^{d_z} to full near-field
wideband THz channel tensor H_hat in R^{2K x N_r x N_t}.
Default: 2K=16 (8 subcarriers real/imag), N_r=32, N_t=256.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DCSChannelGenerator(nn.Module):
    def __init__(self, latent_dim=128, out_channels=16, nr=32, nt=256):
        """
        Args:
            latent_dim (int): Dimension of latent code z (e.g., 128).
            out_channels (int): 2 * K subcarriers (default 16 for K=8 subcarriers).
            nr (int): Number of Rx antennas (default 32).
            nt (int): Number of Tx antennas (default 256).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.nr = nr
        self.nt = nt

        # Initial projection to spatial feature map (4 x 32)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 32),
            nn.BatchNorm1d(256 * 4 * 32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Upsampling Stage 1: (4 x 32) -> (8 x 64)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Upsampling Stage 2: (8 x 64) -> (16 x 128)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Upsampling Stage 3: (16 x 128) -> (32 x 256)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Final Refinement Conv: (32 x 256) -> (32 x 256) with out_channels (16)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh() # Bounded output, scaled to normalized channel range
        )

    def forward(self, z):
        """
        Forward pass from latent code z to estimated channel tensor H.
        Args:
            z (torch.Tensor): Tensor of shape (B, latent_dim).
        Returns:
            torch.Tensor: Channel tensor of shape (B, out_channels, nr, nt).
        """
        B = z.size(0)
        h = self.fc(z).view(B, 256, 4, 32)
        h = self.deconv1(h) # (B, 128, 8, 64)
        h = self.deconv2(h) # (B, 64, 16, 128)
        h = self.deconv3(h) # (B, 32, 32, 256)
        out = self.final_conv(h) # (B, 16, 32, 256)
        return out
