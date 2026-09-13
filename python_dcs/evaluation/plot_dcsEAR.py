import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import norm

# -------------------------
# Repo root + imports
# -------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(THIS_DIR).lower() == "src":
    REPO_DIR = os.path.dirname(THIS_DIR)
    SRC_DIR = THIS_DIR
else:
    REPO_DIR = THIS_DIR
    SRC_DIR = os.path.join(REPO_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import utils
from model import Generator

# -------------------------
# 1) CONFIG
# -------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ⚠️ UPDATE THIS NUMBER AFTER RUNNING THE CALIBRATION SWEEP BELOW ⚠️
SNR_FIXED = 24.8

Nt, Nr = 256, 32
Ns = 4
Nrft, Nrfr = 4, 4
SEED = 417

LATENT_DIM = 100
IMG_SIZE = (16, 32, 256)
K = IMG_SIZE[0] // 2
A = Nt * Nr

rho_vec = np.linspace(0.05, 1.0, 15)
N_TEST = 50
BATCH_SIZE = 25

GD_STEP = 50
N_RESTART = 1
LR_EST = 0.1
BETAS_EST = (0.9, 0.99)

# Required to prevent THz pathloss underflow
NORMALIZE_HEFF = True

CKPT_PATH = os.path.join(REPO_DIR, "ckpt", "dcs_r32t256k8", "e300b50gd10np100dl100.pth.tar")
TRAIN_PT = os.path.join(REPO_DIR, "data", "channel-r32t256k8-n5000.pt")
TEST_PT = os.path.join(REPO_DIR, "data", "channel-r32t256k8-n1000.pt")


# -------------------------
# 2) HELPERS
# -------------------------
def real2complex(Areal: torch.Tensor) -> torch.Tensor:
    C = Areal.shape[1] // 2
    return Areal[:, :C, :, :] + 1j * Areal[:, C:, :, :]


def make_phi_mat(Phi: torch.Tensor):
    def phi_mat(a: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ba, ncaj -> ncbj", Phi, a)

    return phi_mat


def add_awgn_complex(y_clean: torch.Tensor, snr_db: float) -> torch.Tensor:
    sig_pow = torch.mean(torch.abs(y_clean) ** 2)
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_pow = sig_pow / (snr_lin + 1e-12)
    noise_std = torch.sqrt(noise_pow / 2.0)
    n = noise_std * (torch.randn_like(y_clean.real) + 1j * torch.randn_like(y_clean.imag))
    return y_clean + n


def compute_ear_batched(H_true, H_est, snr_db, Np, T_coh, N_streams=4) -> float:
    snr_linear = 10.0 ** (snr_db / 10.0)

    if np.isinf(T_coh):
        eta = 1.0
    else:
        eta = max(0.0, 1.0 - (Np / float(T_coh)))

    U, S, Vh = torch.linalg.svd(H_est, full_matrices=False)
    Ns_use = min(N_streams, U.shape[-1], Vh.shape[-2])
    if Ns_use <= 0:
        return 0.0

    W = U[..., :, :Ns_use]
    F = Vh[..., :Ns_use, :].transpose(-2, -1).conj()

    H_eff = W.transpose(-2, -1).conj() @ H_true @ F  # (B,K,Ns,Ns)

    if NORMALIZE_HEFF:
        pow_ = (H_eff.abs() ** 2).mean(dim=(-1, -2), keepdim=True) + 1e-12
        H_eff = H_eff / torch.sqrt(pow_)

    snr_per_stream = snr_linear / float(Ns_use)
    I = torch.eye(Ns_use, dtype=H_true.dtype, device=H_true.device).view(1, 1, Ns_use, Ns_use)
    C_matrix = I + snr_per_stream * (H_eff.transpose(-2, -1).conj() @ H_eff)

    eigvals = torch.linalg.eigvalsh(C_matrix).real
    eigvals = torch.clamp(eigvals, min=1e-12)
    rate = torch.sum(torch.log2(eigvals), dim=-1)  # (B,K)

    return float(rate.mean().item() * eta)


# -------------------------
# 3) LOAD MODEL + DATA
# -------------------------
print(f"Device: {device}")
print(f"Repo:   {REPO_DIR}")
print(f"Loading checkpoint: {CKPT_PATH}")

ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
gen = Generator(IMG_SIZE, LATENT_DIM).to(device)
gen.load_state_dict(ckpt["generator_state_dict"])
gen.eval()

_, _, mean, std = torch.load(TRAIN_PT, weights_only=False)
std_norm = float(np.linalg.norm(std))

H_all, h_all, _, _ = torch.load(TEST_PT, weights_only=False)
H_all = H_all[:N_TEST].to(device)
h_all = h_all[:N_TEST].to(device)


def z2h(z: torch.Tensor) -> torch.Tensor:
    Gz = gen(z)
    Gz_rev = utils.transform_reverse(Gz, mean, std)
    gz = utils.vectorize(Gz_rev)
    return real2complex(gz)


def loss_inner(y: torch.Tensor, phi_mat, z: torch.Tensor) -> torch.Tensor:
    n_batch = y.shape[0]
    gz = z2h(z)
    err = (norm((y - phi_mat(gz)).reshape(n_batch, -1), dim=1) / (std_norm + 1e-12)) ** 2
    return err


# -------------------------
# 3.5) FIND THE FAIR SNR
# -------------------------
print("\n" + "=" * 50)
print("--- 🔍 FINDING THE FAIR SNR (TARGET: ~21.0 b/s/Hz) ---")
with torch.no_grad():
    h_b_test = h_all[:BATCH_SIZE]
    cur_bch_test = h_b_test.shape[0]
    H_true_test = h_b_test.squeeze(-1).reshape(cur_bch_test, K, Nr, Nt)

    for test_snr in range(10, 26):
        cap = compute_ear_batched(H_true_test, H_true_test, float(test_snr), 0, float("inf"), Ns)
        marker = "  <--- CLOSEST TO 21.0!" if abs(cap - 21.0) < 0.6 else ""
        print(f"If SNR_FIXED = {test_snr:2d} dB  --> Perfect CSI = {cap:.2f} b/s/Hz{marker}")
print("=" * 50 + "\n")
print("⚠️  ACTION REQUIRED: Update 'SNR_FIXED' on line 32 to the dB value that gives ~21.0 b/s/Hz.")
print("Then, run this script again to generate the final CSV!\n")

# -------------------------
# 4) MAIN LOOP
# -------------------------
results_ear_inf, results_ear_1024, results_ear_512, results_ear_perf = [], [], [], []

print(f"Running DCS EAR vs Measurements (SNR={SNR_FIXED} dB, dl={LATENT_DIM})")
debug_done = False

for rho in rho_vec:
    Np = int(round(rho * Nt))
    rho_eff = Np / Nt

    Phi = utils.get_measurement_matrix(Nt, Nr, Np, Ns, Nrft, Nrfr, SEED).to(torch.complex64).to(device)
    phi_mat = make_phi_mat(Phi)

    ear_bch_inf, ear_bch_1024, ear_bch_512, ear_bch_perf = [], [], [], []

    print(f"Np={Np:3d} (rho={rho_eff:.2f}) ...", end=" ", flush=True)

    for start in range(0, N_TEST, BATCH_SIZE):
        end = min(N_TEST, start + BATCH_SIZE)

        h_b = h_all[start:end]
        cur_bch = h_b.shape[0]

        y_clean = phi_mat(h_b)
        y_b = add_awgn_complex(y_clean, SNR_FIXED)

        if not debug_done:
            with torch.no_grad():
                print(f"\n  DEBUG: mean|h|^2={torch.mean(torch.abs(h_b) ** 2).item():.3e}, "
                      f"mean|y_clean|^2={torch.mean(torch.abs(y_clean) ** 2).item():.3e}")
            debug_done = True

        best_z = None
        best_loss = float("inf")

        for _ in range(N_RESTART):
            z = torch.randn((cur_bch, LATENT_DIM), device=device, requires_grad=True)
            z.data /= (z.data.norm(dim=1, keepdim=True) + 1e-12)

            optz = optim.Adam([z], lr=LR_EST, betas=BETAS_EST)

            for _step in range(GD_STEP):
                optz.zero_grad()
                loss_vec = loss_inner(y_b, phi_mat, z)
                loss_vec.sum().backward()
                optz.step()

                with torch.no_grad():
                    z.data /= (z.data.norm(dim=1, keepdim=True) + 1e-12)

            with torch.no_grad():
                cur_loss = loss_inner(y_b, phi_mat, z).mean().item()
                if cur_loss < best_loss:
                    best_loss = cur_loss
                    best_z = z.detach().clone()

        with torch.no_grad():
            H_true = h_b.squeeze(-1).reshape(cur_bch, K, Nr, Nt)
            H_est = z2h(best_z).squeeze(-1).reshape(cur_bch, K, Nr, Nt)

            ear_inf = compute_ear_batched(H_true, H_est, SNR_FIXED, Np, float("inf"), Ns)
            ear_perf = compute_ear_batched(H_true, H_true, SNR_FIXED, 0, float("inf"), Ns)

            ear_bch_inf.append(ear_inf)
            ear_bch_1024.append(compute_ear_batched(H_true, H_est, SNR_FIXED, Np, 1024, Ns))
            ear_bch_512.append(compute_ear_batched(H_true, H_est, SNR_FIXED, Np, 512, Ns))
            ear_bch_perf.append(ear_perf)

    results_ear_inf.append(float(np.mean(ear_bch_inf)))
    results_ear_1024.append(float(np.mean(ear_bch_1024)))
    results_ear_512.append(float(np.mean(ear_bch_512)))
    results_ear_perf.append(float(np.mean(ear_bch_perf)))

    print(f"EAR(Inf)={results_ear_inf[-1]:.6f}  Perfect={results_ear_perf[-1]:.6f}")

# -------------------------
# 5) PLOT + SAVE TO CSV
# -------------------------
x_axis = [int(round(r * Nt)) for r in rho_vec]

plt.figure(figsize=(8, 6))
plt.plot(x_axis, results_ear_perf, "g+-", linewidth=2, markersize=8, label="Perfect CSI")
plt.plot(x_axis, results_ear_inf, "r-", linewidth=2, label=r"DCS ($T_{coh}=\infty$)")
plt.plot(x_axis, results_ear_1024, "r--", linewidth=2, label=r"DCS ($T_{coh}=1024$)")
plt.plot(x_axis, results_ear_512, "r:", linewidth=2, label=r"DCS ($T_{coh}=512$)")

plt.grid(True, linestyle="--", alpha=0.7)
plt.xlabel(r"Number of Measurements ($M_T^{tr}$)", fontsize=12, fontweight="bold")
plt.ylabel(r"$R_{eff}$ (b/s/Hz)", fontsize=12, fontweight="bold")
plt.title(f"DCS EAR vs Measurements (SNR={SNR_FIXED} dB, dl={LATENT_DIM})", fontsize=14)
plt.legend(loc="lower right", fontsize=12)

out_png = os.path.join(REPO_DIR, f"DCS_EAR_vs_Measurements_SNR{int(SNR_FIXED)}_dl{LATENT_DIM}.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print("\nSaved plot:", out_png)

out_csv = os.path.join(REPO_DIR, f"DCS_EAR_vs_Measurements_SNR{int(SNR_FIXED)}_dl{LATENT_DIM}.csv")
arr = np.column_stack([
    np.array(x_axis),
    np.array(results_ear_inf),
    np.array(results_ear_1024),
    np.array(results_ear_512),
    np.array(results_ear_perf),
])
np.savetxt(out_csv, arr, delimiter=",", header="Np,EAR_inf,EAR_1024,EAR_512,EAR_perfect", comments="")
print("Saved CSV :", out_csv)

plt.show()