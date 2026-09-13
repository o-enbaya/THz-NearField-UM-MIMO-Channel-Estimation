"""
THz Channel Dataset Loader with Channel-wise Normalization.
Thesis Reference: Chapter 4 (Eq. 4.12 - 4.14, Section 4.3).

Loads MATLAB generated wideband THz channel matrices, applies
channel-wise normalization across realizations:
    H[:, c, :, :] = (H[:, c, :, :] - mu_c) / sigma_c
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np

class THzChannelDataset(Dataset):
    def __init__(self, data_array=None, mat_path=None, normalize=True):
        """
        Args:
            data_array (numpy.ndarray or torch.Tensor): Pre-loaded array of shape (N, 2K, Nr, Nt).
            mat_path (str): Optional path to .mat channel dataset file.
            normalize (bool): If True, apply Equations 4.12-4.14 normalization.
        """
        if data_array is not None:
            if isinstance(data_array, np.ndarray):
                self.channels = torch.from_numpy(data_array).float()
            else:
                self.channels = data_array.float()
        elif mat_path is not None and os.path.exists(mat_path):
            try:
                import scipy.io as sio
                mat = sio.loadmat(mat_path)
                key = [k for k in mat.keys() if not k.startswith('__')][0]
                arr = mat[key]
                self.channels = torch.from_numpy(arr).float()
            except Exception as e:
                raise RuntimeError(f"Error loading {mat_path}: {e}")
        else:
            # Generate synthetic channel samples for testing
            print("Notice: Initializing synthetic near-field THz channel dataset for demonstration.")
            self.channels = torch.randn(100, 16, 32, 256).float()

        self.normalize = normalize
        if self.normalize:
            self.mu, self.sigma = self._compute_channel_stats(self.channels)
            self.channels = self._apply_normalization(self.channels, self.mu, self.sigma)
        else:
            self.mu = torch.zeros(self.channels.shape[1])
            self.sigma = torch.ones(self.channels.shape[1])

    def _compute_channel_stats(self, H):
        """Compute mu_c and sigma_c across all realizations (Eq 4.13, 4.14)."""
        # H shape: (N, C, Nr, Nt)
        C = H.shape[1]
        mu = torch.zeros(C)
        sigma = torch.zeros(C)
        for c in range(C):
            ch_slice = H[:, c, :, :]
            mu[c] = torch.mean(ch_slice)
            sigma[c] = torch.std(ch_slice, unbiased=True) + 1e-8
        return mu, sigma

    def _apply_normalization(self, H, mu, sigma):
        """Apply Eq. 4.12: (H - mu) / sigma."""
        H_norm = torch.zeros_like(H)
        for c in range(H.shape[1]):
            H_norm[:, c, :, :] = (H[:, c, :, :] - mu[c]) / sigma[c]
        return H_norm

    def denormalize(self, H_norm):
        """Invert Eq. 4.12: H = H_norm * sigma + mu."""
        H = torch.zeros_like(H_norm)
        for c in range(H_norm.shape[1]):
            H[:, c, :, :] = H_norm[:, c, :, :] * self.sigma[c] + self.mu[c]
        return H

    def __len__(self):
        return self.channels.shape[0]

    def __getitem__(self, idx):
        return self.channels[idx]
