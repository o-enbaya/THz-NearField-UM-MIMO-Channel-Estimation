"""
THz Compressive Measurement Operator.
Thesis Reference: Chapter 3 (Eq. 3.1-3.6), Chapter 4 (Section 4.3).

Simulates the hybrid subarray (AoSA) measurement matrix Phi
with 1-bit phase shifters and zero-mean circular symmetric pilots.
"""

import torch
import torch.nn as nn
import numpy as np

class THzMeasurementOperator(nn.Module):
    def __init__(self, nr=32, nt=256, k_sub=8, m_pilots=100, one_bit_phase=True, device='cpu'):
        super().__init__()
        self.nr = nr
        self.nt = nt
        self.k_sub = k_sub
        self.m_pilots = m_pilots
        self.one_bit_phase = one_bit_phase
        self.device = device

        # Construct random measurement matrix Phi (Real representation)
        # Input channel vector size per subcarrier: 2 * (Nr * Nt)
        # Total measurement dimension: M pilots
        in_features = 2 * k_sub * nr * nt
        self.in_features = in_features

        # Random projection matrix
        if one_bit_phase:
            # 1-bit phase quantization: elements are in {-1, +1} / sqrt(M)
            raw_matrix = torch.sign(torch.randn(m_pilots, in_features))
            raw_matrix[raw_matrix == 0] = 1.0
            matrix = raw_matrix / np.sqrt(m_pilots)
        else:
            matrix = torch.randn(m_pilots, in_features) / np.sqrt(m_pilots)

        self.register_buffer('phi', matrix.to(device))

    def forward(self, h_tensor):
        """
        Compress channel tensor H into pilot measurements y.
        Args:
            h_tensor: (B, 2K, Nr, Nt)
        Returns:
            y: (B, M_pilots)
        """
        B = h_tensor.size(0)
        h_flat = h_tensor.view(B, -1)
        y = torch.matmul(h_flat, self.phi.t())
        return y

    def add_noise(self, y, snr_db):
        """Add AWGN noise corresponding to transmit SNR in dB."""
        sig_pwr = torch.mean(y ** 2, dim=-1, keepdim=True)
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_pwr = sig_pwr / snr_linear
        noise = torch.randn_like(y) * torch.sqrt(noise_pwr)
        return y + noise
