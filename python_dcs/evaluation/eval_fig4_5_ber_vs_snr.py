import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

"""
BERvsSNR_Stage3_Eval.py
=========================================
PHYSICS-INFORMED INFERENCE (QPSK PIVOT)

Modifications:
1. MODULATION: Pivoted from 16-QAM to QPSK (4-QAM) to widen decision boundaries
   and accommodate the continuous phase approximation of the neural network.
2. SPATIAL MULTIPLEXING: Maintained Ns=2 to prove the PINN successfully learned
   to orthogonalize the spatial streams.
3. PRECISION TTO: Increased N_RESTART to 10 and lowered LR_INNER to 0.02 for a
   brute-force, fine-grained search of the latent space.
"""

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

import utils
from model import Generator

# =============================================================================
# 1)  PATHS - POINTING TO STAGE 3 WEIGHTS
# =============================================================================
CKPT_PATH = "./ckpt/dcs_stage3_physics/finetuned_stage3_physics_e5.pth.tar"
TEST_PT = "./data/channel-r32t256k8-n1000.pt"
TRAIN_PT = "./data/channel-r32t256k8-n5000.pt"

# =============================================================================
# 2)  SYSTEM PARAMETERS & TTO CONFIGURATION
# =============================================================================
Nt, Nr = 256, 32
K = 8
Np = 256
Ns = 2

N_TEST = 100
BATCH_SIZE = 25
N_FRAMES = 100

B_sys = 10e9
N0_sc = 10 ** ((-173.8 + 10 * np.log10(B_sys) - 30) / 10)


# >>> TTO PRECISION PUSH <<<
def get_T_gd(snr_db):
    if snr_db < -10: return 100
    if snr_db < 0: return 200
    if snr_db < 10: return 400
    if snr_db < 20: return 500
    return 600


N_RESTART = 10  # Brute-force search for the global minimum
LR_INNER = 0.02  # Fine-grained steps to settle perfectly into the valley
BETAS_EST = (0.9, 0.99)
LATENT_DIM = 100
IMG_SIZE = (16, 32, 256)

SNR_VEC = [-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]

SEED = 417
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 3)  LOAD STAGE 3 MODEL & NORMALISATION
# =============================================================================
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
gen = Generator(IMG_SIZE, LATENT_DIM).to(device)
gen.load_state_dict(ckpt["generator_state_dict"])
gen.eval()

H_train_tensor, _, mean_stored, std_stored = torch.load(TRAIN_PT, weights_only=False)


def as_np(x):
    if isinstance(x, torch.Tensor): return x.numpy()
    if isinstance(x, np.ndarray):   return x
    return np.array(x)


mean_s, std_s = as_np(mean_stored), as_np(std_stored)
sn = float(np.linalg.norm(std_s))

if sn < 1e-6:
    H_np = H_train_tensor.numpy()
    mean = H_np.mean(axis=(0, 2, 3))
    std = H_np.std(axis=(0, 2, 3))
    std = np.where(std < 1e-30, 1.0, std)
    sn = float(np.linalg.norm(std))
else:
    mean, std = mean_s, std_s

std_norm = sn

# =============================================================================
# 4)  LOAD TEST DATA & HELPERS
# =============================================================================
H_all, h_all, _, _ = torch.load(TEST_PT, weights_only=False)
H_all = H_all[:N_TEST].to(device)
h_all = h_all[:N_TEST].to(device)


def rc(x):
    C = x.shape[1] // 2
    return x[:, :C, :, :] + 1j * x[:, C:, :, :]


def z2h(z):
    Gz = gen(z)
    Gz_rev = utils.transform_reverse(Gz, mean, std)
    gz = utils.vectorize(Gz_rev)
    C = gz.shape[1] // 2
    return gz[:, :C, :, :] + 1j * gz[:, C:, :, :]


def make_phi(Phi):
    return lambda h: torch.einsum("ba, ncaj -> ncbj", Phi, h)


def cs_loss(y_eff, phi_fn, z):
    B = y_eff.shape[0]
    diff = (y_eff - phi_fn(z2h(z))).reshape(B, -1)
    return (torch.linalg.norm(diff, dim=1) / max(std_norm, 1e-12)) ** 2


def awgn(ref, pwr):
    s = np.sqrt(pwr / 2.0)
    return (s * (torch.randn_like(ref.real) + 1j * torch.randn_like(ref.imag))).to(torch.complex64)


# =============================================================================
# 5)  QPSK (4-QAM) & SYSTEM SETUP
# =============================================================================
_N4 = np.sqrt(2.0)


def mod4(bits):
    # Maps [0,0] -> -1-1j, [1,1] -> 1+1j, etc.
    r = (bits[:, 0] * 2 - 1)
    i = (bits[:, 1] * 2 - 1)
    return (r + 1j * i) / _N4


def demod4(syms):
    # Decides quadrant based on positive/negative real and imaginary parts
    r = (np.asarray(syms).real > 0).astype(int)
    i = (np.asarray(syms).imag > 0).astype(int)
    return np.column_stack((r, i))


Phi = utils.get_measurement_matrix(Nt, Nr, Np, 4, 4, 4, SEED).to(torch.complex64).to(device)
phi_mat = make_phi(Phi)
H_true_np = rc(H_all).cpu().numpy()
gamma = float(np.mean(np.abs(H_true_np) ** 2))

# =============================================================================
# 6)  MAIN BER LOOP
# =============================================================================
ber_dcs, ber_perfect, nmse_list = [], [], []

print("\n" + "=" * 65)
print(f"  BER vs SNR  (QPSK, Ns={Ns}, Stage 3 Physics-Informed PINN)")
print("=" * 65)

for snr_db in SNR_VEC:
    snr_lin = 10 ** (snr_db / 10.)
    tx_scale = np.sqrt(snr_lin * N0_sc / (gamma * Ns))
    T_GD = get_T_gd(snr_db)

    print(f"\n{'─' * 65}\n  SNR={snr_db:+d}dB  T_GD={T_GD}")
    H_est_list = []

    for b0 in tqdm(range(0, N_TEST, BATCH_SIZE), desc=f"  [{snr_db:+d}dB]", leave=False):
        b1 = min(b0 + BATCH_SIZE, N_TEST)
        h_b = h_all[b0:b1];
        B = h_b.shape[0]

        tx_t = torch.tensor(tx_scale, device=device, dtype=torch.float32)
        y_c = phi_mat(h_b)
        y_eff = (tx_t * y_c + awgn(y_c, N0_sc)) / tx_t

        best_z, best_loss = None, float("inf")
        for _r in range(N_RESTART):
            z = torch.randn((B, LATENT_DIM), device=device, requires_grad=True)
            with torch.no_grad():
                z.data /= (z.data.norm(dim=1, keepdim=True) + 1e-12)

            opt_z = optim.Adam([z], lr=LR_INNER, betas=BETAS_EST)

            for _s in range(T_GD):
                opt_z.zero_grad()
                lv = cs_loss(y_eff, phi_mat, z)
                lv.sum().backward()
                opt_z.step()
                with torch.no_grad(): z.data /= (z.data.norm(dim=1, keepdim=True) + 1e-12)

            with torch.no_grad():
                if lv.mean().item() < best_loss:
                    best_loss = lv.mean().item()
                    best_z = z.detach().clone()

        with torch.no_grad():
            Gz = gen(best_z)
            Gz_rev = utils.transform_reverse(Gz, mean, std)
            H_raw = rc(Gz_rev).cpu().numpy()

            gz_complex = z2h(best_z)
            yf = phi_mat(gz_complex).reshape(B, -1)

            ynf = y_eff.reshape(B, -1)
            num = (yf.conj() * ynf).sum(dim=1)
            den = (yf.conj() * yf).sum(dim=1).real + 1e-12
            alpha = np.clip((num / den).cpu().numpy().real, 0.01, 100.).reshape(B, 1, 1, 1)

            H_est_list.append(alpha * H_raw)

    H_est = np.concatenate(H_est_list, axis=0)

    ch_pwr = np.mean(np.abs(H_true_np) ** 2)
    nmse = np.mean(np.abs(H_est - H_true_np) ** 2) / ch_pwr
    nmse_list.append(nmse)

    print(f"  NMSE = {10 * np.log10(nmse):.2f} dB")

    # ── BER (QPSK Evaluation) ────────────────────────────────────────────────
    e_d = e_p = nb = 0
    for c in range(N_TEST):
        for k in range(K):
            Ht = H_true_np[c, k]
            He = H_est[c, k]

            Ut, _, Vht = np.linalg.svd(Ht, full_matrices=False)
            Wp = Ut[:, :Ns]
            Fp = Vht.conj().T[:, :Ns]

            Ue, _, Vhe = np.linalg.svd(He, full_matrices=False)
            Wd = Ue[:, :Ns]
            Fd = Vhe.conj().T[:, :Ns]

            Hp_e = tx_scale * (Wp.conj().T @ Ht @ Fp)
            Hd_e = tx_scale * (Wd.conj().T @ He @ Fd)

            Eq_p = np.linalg.inv(Hp_e.conj().T @ Hp_e + N0_sc * np.eye(Ns)) @ Hp_e.conj().T
            Eq_d = np.linalg.inv(Hd_e.conj().T @ Hd_e + N0_sc * np.eye(Ns)) @ Hd_e.conj().T

            for _ in range(N_FRAMES):
                bits = np.random.randint(0, 2, (Ns, 2))  # QPSK uses 2 bits per symbol
                s = mod4(bits)

                xp = tx_scale * (Fp @ s)
                xd = tx_scale * (Fd @ s)
                n = np.sqrt(N0_sc / 2.) * (np.random.randn(Nr) + 1j * np.random.randn(Nr))

                rp = Wp.conj().T @ (Ht @ xp + n)
                rd = Wd.conj().T @ (Ht @ xd + n)

                e_p += int(np.sum(bits != demod4(Eq_p @ rp)))
                e_d += int(np.sum(bits != demod4(Eq_d @ rd)))
                nb += bits.size

    ber_perfect.append(e_p / nb)
    ber_dcs.append(e_d / nb)
    print(f"  BER_Perfect={e_p / nb:.4e}  BER_DCS={e_d / nb:.4e}")

# =============================================================================
# 7) SAVE
# =============================================================================
df = pd.DataFrame({"SNR_dB": SNR_VEC, "BER_DCS": ber_dcs, "BER_Perfect": ber_perfect, "NMSE_DCS": nmse_list})
df.to_csv("DCS_BER_vs_SNR.csv", index=False)
print("\n  Saved: DCS_BER_vs_SNR.csv")