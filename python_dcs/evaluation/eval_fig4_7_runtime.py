import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import time
import numpy as np
import argparse
import torch
import torch.optim as optim
import torchvision
from tqdm import tqdm

from model import Generator
import utils
from torch import norm

parser = argparse.ArgumentParser()
parser.add_argument("--lr_est", type=float, default=1e-1)
parser.add_argument("--betas_est", type=float, default=(0.9, 0.99))
parser.add_argument("--latent_dim", type=int, default=100)
parser.add_argument("--img_size", type=int, default=(16, 32, 256))
parser.add_argument("--gd_step", type=int, default=20)
parser.add_argument("--n_test", type=int, default=100)
parser.add_argument("--n_restart", type=int, default=1)
opt, _ = parser.parse_known_args()

print(opt)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

ckpt = torch.load(
    "./ckpt/dcs_r32t256k8/e300b50gd10np100dl100.pth.tar",
    map_location=device,
    weights_only=False
)
gen = Generator(opt.img_size, opt.latent_dim).to(device)
gen.load_state_dict(ckpt["generator_state_dict"])
gen.eval()

_, _, mean, std = torch.load("./data/channel-r32t256k8-n5000.pt", weights_only=False)
std_norm = np.linalg.norm(std)

H_all, h_all, _, _ = torch.load("./data/channel-r32t256k8-n1000.pt", weights_only=False)

Phi = utils.get_measurement_matrix(256, 32, 100, 4, 4, 4, 417).to(torch.complex64).to(device)

SNR_vec = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
nmse_avg = torch.zeros((len(SNR_vec)), device=device)
nmse_avg_db = torch.zeros((len(SNR_vec)), device=device)

runtime_per_snr = np.zeros(len(SNR_vec), dtype=np.float64)
runtime_per_sample_per_snr = np.zeros(len(SNR_vec), dtype=np.float64)

def phi_mat(a):
    return torch.einsum('ba, ncaj -> ncbj', Phi, a)

def real2complex(A):
    C = A.shape[1] // 2
    return A[:, :C, :, :] + 1j * A[:, C:, :, :]

def z2h(z, reverse=True):
    Gz = gen(z)
    Gz_rev = utils.transform_reverse(Gz, mean, std)
    if reverse:
        gz = utils.vectorize(Gz_rev)
    else:
        gz = utils.vectorize(Gz)
    return real2complex(gz)

def loss_inner(m_i, z):
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

for snr_idx, snr in enumerate(SNR_vec):
    print(f"\nSNR: {snr}")

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean, std)
    ])

    H_batch = H_all[:opt.n_test].to(device)
    h_batch = h_all[:opt.n_test].to(device)

    test_loader = torch.utils.data.DataLoader(
        utils.Measurement(H_batch, h_batch, Phi, snr, transform=transform),
        batch_size=opt.n_test,
        shuffle=False,
        pin_memory=False,
        num_workers=0
    )

    pbar = tqdm(total=opt.n_restart * opt.gd_step, desc=f"SNR {snr}: [{snr_idx}/{len(SNR_vec)}]")

    snr_start = time.perf_counter()

    for batch, (H, h, y) in enumerate(test_loader):
        H = H.to(device)
        h = h.to(device)
        y = y.to(device)

        cur_bch = H.shape[0]

        z_best = torch.randn([cur_bch, opt.latent_dim], requires_grad=False, device=device)
        err_recon_best = loss_inner(y, z_best).detach()

        for restart in range(opt.n_restart):
            z = torch.randn([cur_bch, opt.latent_dim], requires_grad=True, device=device)
            optimizer_z = optim.Adam([z], lr=opt.lr_est, betas=opt.betas_est)

            for step in range(opt.gd_step):
                optimizer_z.zero_grad()
                err_recon = loss_inner(y, z)
                err_recon.backward(torch.ones_like(err_recon))
                optimizer_z.step()

                with torch.no_grad():
                    z /= z.norm(dim=1, keepdim=True)
                    err_recon = loss_inner(y, z)
                    z_best = z.detach().clone()
                    err_recon_best = err_recon.detach().clone()
                    pbar.update(1)

        nmse = nmse_compute(z_best, h).detach()

    if device.type == "cuda":
        torch.cuda.synchronize()

    snr_end = time.perf_counter()
    pbar.close()

    runtime_per_snr[snr_idx] = snr_end - snr_start
    runtime_per_sample_per_snr[snr_idx] = runtime_per_snr[snr_idx] / opt.n_test

    nmse_avg[snr_idx] = nmse.mean()
    nmse_avg_db[snr_idx] = 10 * torch.log10(nmse_avg[snr_idx])

    print(f"NMSE: {nmse_avg[snr_idx].item():.6f} = {nmse_avg_db[snr_idx].item():.2f} dB")
    print(f"Runtime for this SNR: {runtime_per_snr[snr_idx]:.6f} s")
    print(f"Runtime per sample:   {runtime_per_sample_per_snr[snr_idx]:.6f} s")

print("\n===== DCS Runtime Summary =====")
print("SNR_vec =", SNR_vec)
print("nmse_avg_db =", nmse_avg_db.detach().cpu().numpy())
print("runtime_per_snr =", runtime_per_snr)
print("runtime_per_sample_per_snr =", runtime_per_sample_per_snr)
print("Average runtime per sample across SNRs = %.6f s" % runtime_per_sample_per_snr.mean())

np.savetxt("dcs_nmse_db.csv", nmse_avg_db.detach().cpu().numpy(), delimiter=",")
np.savetxt("dcs_runtime_per_snr.csv", runtime_per_snr, delimiter=",")
np.savetxt("dcs_runtime_per_sample_per_snr.csv", runtime_per_sample_per_snr, delimiter=",")