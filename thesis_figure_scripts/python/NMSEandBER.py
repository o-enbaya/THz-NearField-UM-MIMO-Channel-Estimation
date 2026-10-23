import numpy as np
import argparse
import torch
import torch.optim as optim
import torchvision
from model import Generator
import utils
from torch import norm
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--lr_est", type=float, default=1e-1, help="adam: learning rate in online stage")
parser.add_argument("--betas_est", type=float, default=(0.9, 0.99),
                    help="adam: decay of momentum of gradient in online stage")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=(16, 32, 256), help="size of each image (C,H,W)")
parser.add_argument("--gd_step", type=int, default=20, help="gradient descent steps")
parser.add_argument("--n_test", type=int, default=100, help="number of test data")
parser.add_argument("--n_restart", type=int, default=1, help="number of restart")
opt, unknown = parser.parse_known_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(f"./ckpt/dcs_r32t256k8/e300b50gd10np100dl100.pth.tar", map_location=device, weights_only=False)
gen = Generator(opt.img_size, opt.latent_dim).to(device)
gen.load_state_dict(ckpt['generator_state_dict'])
gen.eval()

_, _, mean, std = torch.load('./data/channel-r32t256k8-n5000.pt', weights_only=False)
std_norm = np.linalg.norm(std)

H_all, h_all, _, _ = torch.load('./data/channel-r32t256k8-n1000.pt', weights_only=False)

Phi = utils.get_measurement_matrix(256, 32, 100, 4, 4, 4, 417).to(torch.complex64)

SNR_vec = [-20,-15,-10,-5,0,5,10,15,20,25,30]

# --- FAIRNESS ALIGNMENT: Match this to the Ns_train in your MATLAB script ---
Ns_train = 1

nmse_avg = torch.zeros((len(SNR_vec)))
nmse_avg_db = torch.zeros((len(SNR_vec)))
ber_avg = torch.zeros((len(SNR_vec)))
ber_perfect = torch.zeros((len(SNR_vec)))

def phi_mat(a):
    return torch.einsum('ba, ncaj -> ncbj', Phi, a)

def real2complex(A):
    C = A.shape[1] // 2
    A_complex = A[:, :C, :, :] + 1j * A[:, C:, :, :]
    return A_complex

def z2h(z, reverse=True):
    Gz = gen(z)
    Gz_rev = utils.transform_reverse(Gz, mean, std)
    if reverse is True:
        gz = utils.vectorize(Gz_rev)
    else:
        gz = utils.vectorize(Gz)
    return real2complex(gz)

def loss_inner(m_i, Phi, z):
    n_batch = m_i.shape[0]
    gz = z2h(z, reverse=True)
    err = (norm((m_i - phi_mat(gz)).reshape(n_batch, -1), dim=1) / std_norm) ** 2
    return err

def nmse_compute(z, h_bch):
    n_batch = z.shape[0]
    gz = z2h(z, reverse=True)
    err = norm((h_bch - gz).reshape(n_batch, -1), dim=1) ** 2
    h_norm = norm(h_bch.reshape(n_batch, -1), dim=1) ** 2
    return err / h_norm

def qam16_modulate(bits):
    device = bits.device
    val_map = torch.tensor([-3.0, -1.0, 3.0, 1.0], device=device)

    idx_real = bits[..., 0] * 2 + bits[..., 1]
    idx_imag = bits[..., 2] * 2 + bits[..., 3]

    real_part = val_map[idx_real.long()]
    imag_part = val_map[idx_imag.long()]

    symbols = real_part + 1j * imag_part
    return symbols / (10.0 ** 0.5)

def qam16_demodulate(symbols):
    device = symbols.device
    s_scaled = symbols * (10.0 ** 0.5)

    constellation = torch.tensor([
        -3 - 3j, -3 - 1j, -3 + 3j, -3 + 1j,
        -1 - 3j, -1 - 1j, -1 + 3j, -1 + 1j,
        3 - 3j, 3 - 1j, 3 + 3j, 3 + 1j,
        1 - 3j, 1 - 1j, 1 + 3j, 1 + 1j
    ], device=device)

    bit_map = torch.tensor([
        [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
        [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
        [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
        [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]
    ], device=device).float()

    dists = torch.abs(s_scaled.unsqueeze(-1) - constellation)
    min_idx = torch.argmin(dists, dim=-1)

    return bit_map[min_idx]


def compute_ber_svd_2stream_paired(h_true, h_est, snr_db, n_bit_iters=313):
    batch_size, n_sub, n_rx, n_tx = h_true.shape
    snr_linear = 10 ** (snr_db / 10.0)
    n_streams = 2  # --- 2-STREAM OVERRIDE ---

    H_t = h_true.reshape(batch_size * n_sub, n_rx, n_tx)
    H_e = h_est.reshape(batch_size * n_sub, n_rx, n_tx)

    # DCS receiver from estimated channel
    U_e, _, Vh_e = torch.linalg.svd(H_e, full_matrices=False)
    P_dcs = Vh_e.conj().transpose(1, 2)[:, :, :n_streams]
    W_dcs = U_e.conj().transpose(1, 2)[:, :n_streams, :]

    # Perfect CSI receiver from true channel
    U_t, _, Vh_t = torch.linalg.svd(H_t, full_matrices=False)
    P_perf = Vh_t.conj().transpose(1, 2)[:, :, :n_streams]
    W_perf = U_t.conj().transpose(1, 2)[:, :n_streams, :]

    # =====================================================================
    # 2-STREAM MATRIX MMSE EQUALIZATION (Fixed Total Power Constraint)
    # =====================================================================
    H_eff_dcs = torch.bmm(W_dcs, torch.bmm(H_t, P_dcs))
    H_eff_perf = torch.bmm(W_perf, torch.bmm(H_t, P_perf))

    # --- Shared noise variance scaled for total transmit power ---
    # Python equivalent of norm(H, 'fro')^2
    fro_sq_perf = torch.sum(torch.abs(H_eff_perf) ** 2, dim=(1, 2), keepdim=True)
    noise_var_common = (fro_sq_perf / (n_streams ** 2)) / snr_linear
    noise_var_common[noise_var_common == 0] = 1e-12

    I2 = torch.eye(n_streams, device=h_true.device).unsqueeze(0)

    # Full MIMO Matrix Inversion Formula: H^H * (H*H^H + sigma^2 * I)^-1
    temp_dcs = torch.bmm(H_eff_dcs, H_eff_dcs.conj().transpose(1, 2)) + noise_var_common * I2
    Eq_dcs = torch.bmm(H_eff_dcs.conj().transpose(1, 2), torch.linalg.inv(temp_dcs))

    temp_perf = torch.bmm(H_eff_perf, H_eff_perf.conj().transpose(1, 2)) + noise_var_common * I2
    Eq_perf = torch.bmm(H_eff_perf.conj().transpose(1, 2), torch.linalg.inv(temp_perf))
    # =====================================================================

    total_errors_dcs = 0
    total_errors_perf = 0
    total_bits = 0

    bit_pbar = tqdm(total=n_bit_iters, desc="   Pumping Bits", leave=False)

    for _ in range(n_bit_iters):
        bits = torch.randint(0, 2, (batch_size * n_sub, n_streams, 4), device=h_true.device).float()
        s = qam16_modulate(bits).unsqueeze(-1)  # shape: (B, 2, 1)

        # --- Enforce fixed total transmit power ---
        s_flat = s / (n_streams ** 0.5)

        # Shared normalized complex Gaussian draw
        n0 = (torch.randn(batch_size * n_sub, n_rx, 1, device=h_true.device) +
              1j * torch.randn(batch_size * n_sub, n_rx, 1, device=h_true.device)) / (2.0 ** 0.5)

        # DCS Data Transmission
        y_dcs = torch.bmm(H_t, torch.bmm(P_dcs, s_flat)) + torch.sqrt(noise_var_common) * n0
        rx_sig_dcs = torch.bmm(W_dcs, y_dcs)

        # Detection (Applying the 2x2 MMSE Matrix)
        s_hat_dcs = torch.bmm(Eq_dcs, rx_sig_dcs)
        s_hat_dcs = s_hat_dcs * (n_streams ** 0.5)  # Inverse power scaling

        # Perfect CSI Data Transmission
        y_perf = torch.bmm(H_t, torch.bmm(P_perf, s_flat)) + torch.sqrt(noise_var_common) * n0
        rx_sig_perf = torch.bmm(W_perf, y_perf)

        # Detection (Applying the 2x2 MMSE Matrix)
        s_hat_perf = torch.bmm(Eq_perf, rx_sig_perf)
        s_hat_perf = s_hat_perf * (n_streams ** 0.5)  # Inverse power scaling

        # Demodulation
        bits_hat_dcs = qam16_demodulate(s_hat_dcs.squeeze(-1))
        bits_hat_perf = qam16_demodulate(s_hat_perf.squeeze(-1))

        total_errors_dcs += (bits != bits_hat_dcs).sum().item()
        total_errors_perf += (bits != bits_hat_perf).sum().item()
        total_bits += bits.numel()
        bit_pbar.update(1)

    bit_pbar.close()
    return total_errors_dcs, total_errors_perf, total_bits


for snr_idx, snr in enumerate(SNR_vec):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean, std)
    ])

    Phi = Phi.to(device)
    H_batch = H_all[:opt.n_test].to(device)
    h_batch = h_all[:opt.n_test].to(device)

    # --- THE PILOT AVERAGING UPDATE ---
    # Boosts the measurement SNR by ~6 dB if Ns_train = 4
    effective_est_snr = snr + 10 * np.log10(Ns_train)

    # Use the boosted SNR to generate the pilot measurements (y)
    TestDataloader = torch.utils.data.DataLoader(
        utils.Measurement(H_batch, h_batch, Phi, effective_est_snr, transform=transform),
        batch_size=opt.n_test,
        shuffle=False,
        pin_memory=False,
        num_workers=0
    )

    restart, step = 0, 0
    pbar = tqdm(total=opt.n_restart * opt.gd_step, desc=f'SNR {snr:3d} Channel Est: [{snr_idx + 1}/{len(SNR_vec)}]')

    for batch, (H, h, y) in enumerate(TestDataloader):
        H, y, h, Phi = H.to(device), y.to(device), h.to(device), Phi.to(device)
        cur_bch = H.shape[0]
        z_best = torch.randn([cur_bch, opt.latent_dim], requires_grad=False, device=device)
        err_recon_best = loss_inner(y, Phi, z_best).detach()

        for restart in range(opt.n_restart):
            z = torch.randn([cur_bch, opt.latent_dim], requires_grad=True, device=device)
            optimizer_z = optim.Adam([z], lr=opt.lr_est, betas=opt.betas_est)

            for step in range(opt.gd_step):
                optimizer_z.zero_grad()
                err_recon = loss_inner(y, Phi, z)
                err_recon.backward(torch.ones_like(err_recon))
                optimizer_z.step()

                with torch.no_grad():
                    z /= z.norm(dim=1, keepdim=True)
                    err_recon = loss_inner(y, Phi, z)
                    z_best = z.detach().clone()
                    pbar.update(1)

        nmse = nmse_compute(z_best, h).detach()

    pbar.close()

    with torch.no_grad():
        h_est = z2h(z_best, reverse=True)
        cur_bch = h.shape[0]
        n_subcarriers = 8
        h_reshaped = h.reshape(cur_bch, n_subcarriers, 32, 256)
        h_est_reshaped = h_est.reshape(cur_bch, n_subcarriers, 32, 256)

        # The BER loop still faces the RAW unboosted data SNR
        # ---> UPDATED CALL <---
        errors_dcs, errors_perf, total_bits = compute_ber_svd_2stream_paired(
            h_reshaped, h_est_reshaped, snr, n_bit_iters=313
        )
        ber_avg[snr_idx] = errors_dcs / total_bits
        ber_perfect[snr_idx] = errors_perf / total_bits

    nmse_avg[snr_idx] = nmse.mean()
    nmse_avg_db[snr_idx] = 10 * torch.log10(nmse_avg[snr_idx])

    print(f"NMSE: {nmse_avg[snr_idx]:.4f} = {nmse_avg_db[snr_idx]:.2f} dB | DCS BER: {ber_avg[snr_idx]:.6e} | Perfect CSI BER: {ber_perfect[snr_idx]:.6e}")

print("\n--- FINAL RESULTS ---")
print("NMSE (Linear):", nmse_avg.numpy())
print("NMSE (dB):    ", nmse_avg_db.numpy())

np.set_printoptions(formatter={'float': '{: 0.6e}'.format})
print("16-QAM BER (DCS):         ", ber_avg.numpy())
print("16-QAM BER (Perfect CSI): ", ber_perfect.numpy())
np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan', precision=8, suppress=False, threshold=1000, formatter=None)

# ------------------------------------------------------------------------------
# Arrays explicitly formatted for easy copy-pasting to MATLAB
# ------------------------------------------------------------------------------
print("\n--- ARRAYS FOR MATLAB PLOTTING ---")
print("dcs_raw_results  = [" + ", ".join([f"{x:e}" for x in ber_avg.numpy()]) + "];")
print("perf_raw_results = [" + ", ".join([f"{x:e}" for x in ber_perfect.numpy()]) + "];")
print("----------------------------------\n")

print("Generating and saving plots...")

snr_np = np.array(SNR_vec)
nmse_db_np = nmse_avg_db.numpy()
ber_np = ber_avg.numpy()
ber_perf_np = ber_perfect.numpy()

# Clamp to 1e-6 so the log plot line stretches perfectly to the academic floor
ber_np_plot = np.where(ber_np == 0, 1e-6, ber_np)
ber_perf_plot = np.where(ber_perf_np == 0, 1e-6, ber_perf_np)

plt.figure(figsize=(8, 6))
plt.plot(snr_np, nmse_db_np, marker='o', linestyle='-', color='b', linewidth=2, markersize=8,
         label='DCS Estimated Channel')
plt.xlabel('SNR (dB)', fontsize=12, fontweight='bold')
plt.ylabel('NMSE (dB)', fontsize=12, fontweight='bold')
plt.title('Normalized Mean Square Error vs SNR', fontsize=14, fontweight='bold')
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('nmse_vs_snr.png', dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
# Plot Perfect CSI native curve first
plt.semilogy(snr_np, ber_perf_plot, marker='+', linestyle='-', color='g', linewidth=2.5, markersize=8,
             label='Perfect CSI (Native)')
# Plot DCS curve
plt.semilogy(snr_np, ber_np_plot, marker='s', linestyle='-', color='r', linewidth=2, markersize=8,
             label='Deep CS (Proposed)')

plt.xlabel('SNR (dB)', fontsize=12, fontweight='bold')
plt.ylabel('Bit Error Rate (BER)', fontsize=12, fontweight='bold')
plt.title('BER vs SNR Performance (Pure Beamforming)', fontsize=14, fontweight='bold')
plt.ylim(bottom=1e-6, top=1)
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('ber_vs_snr_1stream.png', dpi=300)
plt.close()

print("Successfully saved 'nmse_vs_snr.png' and 'ber_vs_snr_1stream.png'.")