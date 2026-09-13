import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import os
import csv
import math
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import utils
    from model import Generator
except ImportError:
    from src import utils
    from src.model import Generator

# =========================================================
# DYNAMIC RELATIVE PATHS
# =========================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT_PATH = os.path.join(BASE_DIR, "checkpoints", "dcs_r32t256k8", "e300b50gd10np100dl100.pth.tar")
TRAIN_PT = os.path.join(BASE_DIR, "data", "channel-r32t256k8-n5000.pt")
TEST_PATTERN = os.path.join(BASE_DIR, "data", "channel-r32t256k8-n1000d{dist_str}delta0.0005Delta0.01theta0.523599-Multipath-universal.pt")
OUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "results", "data"))

# =========================================================
# SETTINGS
# =========================================================
DISTANCES = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
SNR_DB = 10.0
N_TEST = 1000

# Must match checkpoint: e300b50gd10np100dl100
LATENT_DIM = 100
IMG_SIZE = (16, 32, 256)
N_P = 100
GD_STEP = 100

N_RESTART = 3
LR_EST = 0.1
BETAS_EST = (0.9, 0.99)
SEED = 417

# Small batch for 6 GB GPU
EVAL_BATCH_SIZE = 2
# =========================================================


def dist_to_str(dist: float) -> str:
    return f"{dist:g}"


def real2complex(A: torch.Tensor) -> torch.Tensor:
    # [N, 2K, W, H] -> [N, K, W, H]
    C = A.shape[1] // 2
    return A[:, :C, :, :] + 1j * A[:, C:, :, :]


def add_noise_by_snr(m_clean: torch.Tensor, snr_db: float) -> torch.Tensor:
    flat = m_clean.reshape(m_clean.shape[0], -1)
    signal_power = flat.abs().pow(2).mean(dim=1, keepdim=True)
    snr_lin = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_lin
    noise_std = torch.sqrt(noise_power / 2.0)

    noise = (torch.randn_like(flat.real) + 1j * torch.randn_like(flat.real)) * noise_std
    return (flat + noise).reshape_as(m_clean)


def nmse_compute(h_est: torch.Tensor, h_true: torch.Tensor) -> torch.Tensor:
    n_batch = h_true.shape[0]
    err = torch.linalg.norm((h_true - h_est).reshape(n_batch, -1), dim=1) ** 2
    h_norm = torch.linalg.norm(h_true.reshape(n_batch, -1), dim=1) ** 2
    return err / h_norm


def load_train_stats(device):
    if not os.path.exists(TRAIN_PT):
        raise FileNotFoundError(f"Training .pt not found:\n{TRAIN_PT}")

    H_all, h_all, mean, std = torch.load(TRAIN_PT, map_location="cpu", weights_only=False)

    mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(std, dtype=torch.float32, device=device)
    std_norm = float(torch.linalg.norm(std).item())

    del H_all, h_all
    return mean, std, std_norm


def build_generator(device):
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found:\n{CKPT_PATH}")

    gen = Generator(IMG_SIZE, LATENT_DIM).to(device)
    checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    gen.load_state_dict(checkpoint["generator_state_dict"])
    gen.eval()
    return gen


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Checkpoint does not contain mean/std. Falling back to TRAIN_PT...")

    mean, std, std_norm = load_train_stats(device)
    gen = build_generator(device)

    # Same measurement matrix call as training
    Phi = utils.get_measurement_matrix(256, 32, N_P, 4, 4, 4, 417).to(device).to(torch.complex64)

    def phi_mat(a):
        # EXACTLY like dcs-train.py
        return torch.einsum('ba, ncaj -> ncbj', Phi, a)

    def z2h(z, reverse=True):
        # EXACTLY like dcs-train.py
        Gz = gen(z)
        Gz_rev = utils.transform_reverse(Gz, mean, std)
        if reverse:
            gz = utils.vectorize(Gz_rev)
        else:
            gz = utils.vectorize(Gz)
        return real2complex(gz)

    results = []

    for dist in DISTANCES:
        dist_str = dist_to_str(dist)
        test_path = TEST_PATTERN.format(dist_str=dist_str)

        if not os.path.exists(test_path):
            print(f"[SKIP] Missing file:\n{test_path}")
            continue

        print("\n" + "=" * 80)
        print(f"Distance: {dist_str} m")
        print(f"File    : {test_path}")

        H_all, h_all, _, _ = torch.load(test_path, map_location="cpu", weights_only=False)
        n_use = min(N_TEST, h_all.shape[0])

        nmse_sum = 0.0
        n_done = 0

        for start in tqdm(range(0, n_use, EVAL_BATCH_SIZE), desc=f"d={dist_str}m", leave=False):
            end = min(start + EVAL_BATCH_SIZE, n_use)
            h_true = h_all[start:end].to(device).to(torch.complex64)
            bs = h_true.shape[0]

            with torch.no_grad():
                m_clean = phi_mat(h_true)
                m_noisy = add_noise_by_snr(m_clean, SNR_DB)

            best_recon_err = torch.full((bs,), float("inf"), device=device)
            best_h_est = None

            for _ in range(N_RESTART):
                z = torch.randn([bs, LATENT_DIM], device=device)
                z = z / z.norm(dim=1, keepdim=True)
                z.requires_grad_(True)

                opt_z = optim.Adam([z], lr=LR_EST, betas=BETAS_EST)

                for _ in range(GD_STEP):
                    opt_z.zero_grad()

                    h_est_raw = z2h(z, reverse=True)
                    m_est_raw = phi_mat(h_est_raw).reshape(bs, -1)
                    m_noisy_flat = m_noisy.reshape(bs, -1)

                    # LS complex gain/phase calibration in measurement domain
                    num = (m_noisy_flat * m_est_raw.conj()).sum(dim=1, keepdim=True)
                    den = m_est_raw.abs().pow(2).sum(dim=1, keepdim=True) + 1e-12
                    c = num / den

                    diff = m_noisy_flat - (m_est_raw * c)
                    recon_err = (torch.linalg.norm(diff, dim=1) / std_norm) ** 2

                    recon_err.backward(torch.ones_like(recon_err))
                    opt_z.step()

                    with torch.no_grad():
                        z /= z.norm(dim=1, keepdim=True)

                with torch.no_grad():
                    h_est_raw = z2h(z, reverse=True)
                    m_est_raw = phi_mat(h_est_raw).reshape(bs, -1)
                    m_noisy_flat = m_noisy.reshape(bs, -1)

                    num = (m_noisy_flat * m_est_raw.conj()).sum(dim=1, keepdim=True)
                    den = m_est_raw.abs().pow(2).sum(dim=1, keepdim=True) + 1e-12
                    c = num / den

                    # Apply the same scalar to the channel estimate
                    h_est = h_est_raw * c.view(bs, 1, 1, 1)

                    diff = m_noisy_flat - (m_est_raw * c)
                    recon_err = (torch.linalg.norm(diff, dim=1) / std_norm) ** 2

                    if best_h_est is None:
                        best_recon_err = recon_err.clone()
                        best_h_est = h_est.clone()
                    else:
                        better = recon_err < best_recon_err
                        best_recon_err[better] = recon_err[better]
                        best_h_est[better] = h_est[better]

            with torch.no_grad():
                nmse = nmse_compute(best_h_est, h_true)
                nmse_sum += nmse.sum().item()
                n_done += bs

            del h_true, m_clean, m_noisy, best_recon_err, best_h_est, z
            torch.cuda.empty_cache()

        del H_all, h_all
        torch.cuda.empty_cache()

        if n_done == 0:
            continue

        nmse_mean = nmse_sum / n_done
        nmse_db = 10 * math.log10(max(nmse_mean, 1e-12))

        print(f"Avg CALIBRATED NMSE: {nmse_db:.4f} dB")

        results.append({
            "distance_m": dist,
            "snr_db": SNR_DB,
            "n_test": n_done,
            "nmse_linear": nmse_mean,
            "nmse_db": nmse_db,
        })

    if not results:
        print("No results produced.")
        return

    csv_path = os.path.join(OUT_DIR, "nmse_vs_distance_ls_calibrated.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["distance_m", "snr_db", "n_test", "nmse_linear", "nmse_db"]
        )
        writer.writeheader()
        writer.writerows(results)

    dists = [r["distance_m"] for r in results]
    nmse_db_vals = [r["nmse_db"] for r in results]

    plt.figure(figsize=(7, 5))
    plt.plot(dists, nmse_db_vals, marker="o")
    plt.xlabel("Distance (m)")
    plt.ylabel("Calibrated NMSE (dB)")
    plt.title(f"DCS Calibrated NMSE vs Distance @ {SNR_DB:g} dB")
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(OUT_DIR, "nmse_vs_distance_ls_calibrated.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print("\n" + "=" * 80)
    print(f"Saved CSV : {csv_path}")
    print(f"Saved plot: {plot_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()