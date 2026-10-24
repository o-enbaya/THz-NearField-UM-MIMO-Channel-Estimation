import os
import numpy as np
import torch
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from torch import norm
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from model import Generator

# -------------------------
# 1) CONFIG
# -------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SNR_FIXED = 10.0
Nt, Nr = 256, 32
Ns, Nrft, Nrfr = 4, 4, 4
SEED = 417

LATENT_DIM = 100
IMG_SIZE = (16, 32, 256)

# Sweeping from 0.05 to 1.0 using 20 points to match baseline algorithms
rho_vec = np.linspace(0.05, 1.0, 20)

N_TEST = 50          # how many test channels to average
BATCH_SIZE = 25

GD_STEP = 50         # IMPORTANT: DCS uses small steps (paper + dcs-test.py)
N_RESTART = 3
LR_EST = 0.1
BETAS_EST = (0.9, 0.99)

CKPT_PATH = "./ckpt/dcs_r32t256k8/e300b50gd10np100dl100.pth.tar"
TRAIN_PT = "./data/channel-r32t256k8-n5000.pt"
TEST_PT  = "./data/channel-r32t256k8-n1000.pt"

# -------------------------
# 2) HELPERS
# -------------------------
def real2complex(A):
    C = A.shape[1] // 2
    return A[:, :C, :, :] + 1j * A[:, C:, :, :]

# -------------------------
# 3) LOAD MODEL + DATA
# -------------------------
print(f"Device: {device}")
print(f"Loading checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)

gen = Generator(IMG_SIZE, LATENT_DIM).to(device)
gen.load_state_dict(ckpt["generator_state_dict"])
gen.eval()

# mean/std from training pt
_, _, mean, std = torch.load(TRAIN_PT, weights_only=False)
std_norm = float(np.linalg.norm(std))

transform = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(mean, std)
])

# test data
H_all, h_all, _, _ = torch.load(TEST_PT, weights_only=False)

# take first N_TEST
H_all = H_all[:N_TEST].to(device)
h_all = h_all[:N_TEST].to(device)

def z2h(z, reverse=True):
    Gz = gen(z)
    Gz_rev = utils.transform_reverse(Gz, mean, std)
    if reverse:
        gz = utils.vectorize(Gz_rev)
    else:
        gz = utils.vectorize(Gz)
    return real2complex(gz)

def make_phi_mat(Phi):
    def phi_mat(a):
        return torch.einsum("ba, ncaj -> ncbj", Phi, a)
    return phi_mat

def loss_inner(y, phi_mat, z):
    n_batch = y.shape[0]
    gz = z2h(z, reverse=True)
    err = (norm((y - phi_mat(gz)).reshape(n_batch, -1), dim=1) / (std_norm + 1e-12)) ** 2
    return err

def nmse_compute(z, h_true):
    n_batch = z.shape[0]
    gz = z2h(z, reverse=True)
    err = norm((h_true - gz).reshape(n_batch, -1), dim=1) ** 2
    h_norm = norm(h_true.reshape(n_batch, -1), dim=1) ** 2
    return err / (h_norm + 1e-12)

# -------------------------
# 4) LOOP OVER rho (sweep Np)
# -------------------------
results_nmse_db = []

print(f"\nRunning DCS NMSE vs rho (dl={LATENT_DIM}, SNR={SNR_FIXED} dB)")
for rho in rho_vec:
    Np = int(round(rho * Nt))
    rho_eff = Np / Nt

    Phi = utils.get_measurement_matrix(Nt, Nr, Np, Ns, Nrft, Nrfr, SEED).to(torch.complex64).to(device)
    phi_mat = make_phi_mat(Phi)

    test_loader = DataLoader(
        utils.Measurement(H_all, h_all, Phi, SNR_FIXED, transform=transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    nmse_list = []

    print(f"rho={rho_eff:.3f}  (Np={Np}, rows={Phi.shape[0]}=Ns*Np)", end=" ... ", flush=True)

    for (H_b, h_b, y_b) in test_loader:
        H_b = H_b.to(device)
        h_b = h_b.to(device)
        y_b = y_b.to(device)

        cur_bch = H_b.shape[0]

        z_best = torch.randn((cur_bch, LATENT_DIM), device=device)
        z_best = z_best / (z_best.norm(dim=1, keepdim=True) + 1e-12)
        best_err = loss_inner(y_b, phi_mat, z_best).detach()

        for _ in range(N_RESTART):
            z = torch.randn((cur_bch, LATENT_DIM), device=device, requires_grad=True)
            z.data = z.data / (z.data.norm(dim=1, keepdim=True) + 1e-12)

            optz = optim.Adam([z], lr=LR_EST, betas=BETAS_EST)

            for _step in range(GD_STEP):
                optz.zero_grad()
                err = loss_inner(y_b, phi_mat, z)
                err.sum().backward()
                optz.step()

                with torch.no_grad():
                    z.data = z.data / (z.data.norm(dim=1, keepdim=True) + 1e-12)

            with torch.no_grad():
                err_final = loss_inner(y_b, phi_mat, z)
                improved = err_final < best_err
                if improved.any():
                    best_err[improved] = err_final[improved]
                    z_best[improved] = z.detach()[improved]

        with torch.no_grad():
            nmse = nmse_compute(z_best, h_b).detach().cpu().numpy().tolist()
        nmse_list.extend(nmse)

    avg_nmse = float(np.mean(nmse_list))
    nmse_db = 10 * np.log10(avg_nmse + 1e-12)
    results_nmse_db.append(nmse_db)
    print(f"NMSE={nmse_db:.2f} dB")

# -------------------------
# 5) PLOT AND EXPORT CSV
# -------------------------
x_axis = [int(round(r*Nt))/Nt for r in rho_vec]

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(x_axis, results_nmse_db, marker="^", linewidth=2, markersize=8, color="#D95319",
         label=r"DCS ($d_\ell = 100$)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.xlabel(r"Compression Ratio ($\rho = N_p/N_t$)", fontsize=12, fontweight="bold")
plt.ylabel("NMSE (dB)", fontsize=12, fontweight="bold")
plt.title(f"NMSE of DCS vs Compression Ratio (SNR={SNR_FIXED} dB)", fontsize=14)
plt.legend(loc="lower left", fontsize=12)

out_png = f"DCS_NMSE_vs_Rho_SNR{int(SNR_FIXED)}.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"\nSaved plot: {out_png}")

# Export to CSV for MATLAB
out_csv = f"DCS_NMSE_vs_Rho_SNR{int(SNR_FIXED)}.csv"
arr = np.column_stack([np.array(x_axis), np.array(results_nmse_db)])
np.savetxt(out_csv, arr, delimiter=",", header="rho,NMSE_dB", comments="")
print(f"Saved CSV : {out_csv}\n")

plt.show()