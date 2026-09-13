import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import numpy as np
import matplotlib.pyplot as plt
import utils
from model import Generator
from torch.utils.data import TensorDataset, DataLoader

# --- 1. CONFIGURATION ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SNR_FIXED = 40.0
Nt = 256
LATENT_DIM = 100
rho_vec = np.linspace(0.05, 0.6, 12)
results_nmse_db = []


# --- 2. HELPER FUNCTIONS ---
def real2complex(A):
    C = A.shape[1] // 2
    return A[:, :C, :, :] + 1j * A[:, C:, :, :]


def phi_mat(Phi, vec):
    # vec: (Batch, Channels, 8192) | Phi: (Np, 8192) -> Output: (Batch, Channels, Np)
    return torch.einsum('pa, bca -> bcp', Phi, vec)


def add_noise(y, snr_db):
    sig_pwr = torch.mean(torch.abs(y) ** 2)
    noise_pwr = sig_pwr / (10 ** (snr_db / 10))
    noise_std = torch.sqrt(noise_pwr / 2)
    noise = torch.randn_like(y.real) * noise_std + 1j * torch.randn_like(y.imag) * noise_std
    return y + noise


# --- 3. LOAD MODEL & DATA ---
gen = Generator((16, 32, 256), LATENT_DIM).to(device)
ckpt_path = "./ckpt/dcs_r32t256k8/e130b50gd10np100dl100.pth.tar"
print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
gen.load_state_dict(ckpt['generator_state_dict'])
gen.eval()

print("Loading channel data...")
H_all, h_all = torch.load('./data/channel-r32t256k8-n1000.pt', weights_only=False)[:2]
H_test = H_all[:50].to(device)
h_test = h_all[:50].to(device)
dataset = TensorDataset(H_test, h_test)
loader = DataLoader(dataset, batch_size=25, shuffle=False)

# --- 4. SIMULATION LOOP OVER RHO ---
print(f"Starting DCS Simulation over Rho (SNR = {SNR_FIXED} dB)...")
for rho in rho_vec:
    Np = int(rho * Nt)
    print(f"Testing Rho = {rho:.2f} (Pilots = {Np})...", end=" ", flush=True)

    Phi = utils.get_measurement_matrix(256, 32, Np, 4, 4, 4, 417).to(device).to(torch.complex64)
    nmse_batch_list = []

    for H_b, h_b in loader:
        # ---------------------------------------------------------
        # GROUND TRUTH PIPELINE: Permute and Flatten
        # ---------------------------------------------------------
        h_c = h_b
        h_perm = h_c.permute(0, 1, 3, 2)
        h_c_vec = h_perm.reshape(h_perm.shape[0], h_perm.shape[1], -1)

        y_clean = phi_mat(Phi, h_c_vec)
        y_noisy = add_noise(y_clean, SNR_FIXED)

        z = torch.randn([h_b.shape[0], LATENT_DIM], requires_grad=True, device=device)
        optimizer = torch.optim.Adam([z], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)

        best_z = z.clone().detach()
        best_loss = float('inf')

        for step in range(500):
            optimizer.zero_grad()

            # ---------------------------------------------------------
            # GENERATOR PIPELINE: Permute and Flatten EXACTLY the same way
            # ---------------------------------------------------------
            h_hat_c = real2complex(gen(z))
            h_hat_perm = h_hat_c.permute(0, 1, 3, 2)
            h_hat_vec = h_hat_perm.reshape(h_hat_perm.shape[0], h_hat_perm.shape[1], -1)

            y_hat = phi_mat(Phi, h_hat_vec)

            # Loss Function with Z-Prior Regularization
            loss_measure = torch.mean(torch.abs(y_noisy - y_hat) ** 2)
            loss_prior = 0.01 * torch.mean(z ** 2)  # Stops the Generator from exploding
            loss = loss_measure + loss_prior

            loss.backward()
            torch.nn.utils.clip_grad_norm_([z], max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # Track purely the measurement loss for the "best" state
            if loss_measure.item() < best_loss:
                best_loss = loss_measure.item()
                best_z = z.clone().detach()

        # Calculate NMSE purely on the identical flattened vectors
        final_h_hat_c = real2complex(gen(best_z))
        final_h_hat_perm = final_h_hat_c.permute(0, 1, 3, 2)
        final_h_hat_vec = final_h_hat_perm.reshape(final_h_hat_perm.shape[0], final_h_hat_perm.shape[1], -1)

        diff_h = final_h_hat_vec - h_c_vec

        numerator = torch.sum(torch.abs(diff_h) ** 2, dim=[1, 2])
        denominator = torch.sum(torch.abs(h_c_vec) ** 2, dim=[1, 2])

        nmse_samples = (numerator / denominator).cpu().detach().tolist()
        nmse_batch_list.extend(nmse_samples)

    avg_nmse = np.mean(nmse_batch_list)
    val_db = 10 * np.log10(avg_nmse)
    results_nmse_db.append(val_db)
    print(f"NMSE = {val_db:.2f} dB")

# --- 5. PLOTTING THE FIGURE ---
print("Plotting results...")
plt.figure(figsize=(8, 6))
plt.plot(rho_vec, results_nmse_db, 'g^-', linewidth=2, markersize=8, label='DCS ($d_l = 100$)')

plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Compression Ratio ($\\rho$)', fontsize=12, fontweight='bold')
plt.ylabel('NMSE (dB)', fontsize=12, fontweight='bold')
plt.title('NMSE of DCS vs Compression Ratio', fontsize=14)
plt.legend(loc='lower left', fontsize=12)

plt.savefig('DCS_NMSE_vs_Rho.png', dpi=300)
plt.show()