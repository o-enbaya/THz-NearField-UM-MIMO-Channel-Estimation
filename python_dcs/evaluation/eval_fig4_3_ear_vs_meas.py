import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import argparse
import torch
import torch.optim as optim
import torchvision
from model import Generator
import utils
import utils_ear  # Your new EAR functions
from torch.linalg import norm
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--lr_est", type=float, default=1e-1)
parser.add_argument("--betas_est", type=float, default=(0.9, 0.99))
parser.add_argument("--latent_dim", type=int, default=100)
parser.add_argument("--img_size", type=int, default=(16, 32, 256))
parser.add_argument("--gd_step", type=int, default=10)
parser.add_argument("--n_test", type=int, default=100)
parser.add_argument("--n_restart", type=int, default=1)
opt, _ = parser.parse_known_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- EAR EXPERIMENT SETUP ---
fixed_snr = 0  # High SNR for evaluating EAR bounds
measurement_list = [100]  # MUST have trained checkpoints for these!
T_coh_list = [float('inf'), 1024, 512]
results_ear = {t: [] for t in T_coh_list}

# Load testing dataset once
H_all, h_all, _, _ = torch.load('./data/channel-r32t256k8-n1000.pt', weights_only=False)
_, _, mean, std = torch.load('./data/channel-r32t256k8-n5000.pt', weights_only=False)
transform = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean, std)])

for n_p in measurement_list:
    print(f"\n=============================================")
    print(f"Evaluating EAR for Measurements (M) = {n_p}")

    # 1. Generate measurement matrix specific to this M
    Phi = utils.get_measurement_matrix(256, 32, n_p, 4, 4, 4, 417).to(torch.complex64).to(device)

    # 2. Load the specific model trained for this M
    # IMPORTANT: Ensure this filename matches exactly how dcs-train.py saves it
    ckpt_path = f"./ckpt/dcs_r32t256k8/e130b50gd10np{n_p}dl{opt.latent_dim}.pth.tar"
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"SKIPPING: Checkpoint not found for np={n_p}. Run 'python dcs-train.py --n_p {n_p}' first.")
        continue

    gen = Generator(opt.img_size, opt.latent_dim).to(device)
    gen.load_state_dict(ckpt['generator_state_dict'])
    gen.eval()


    # Helper functions bound to this loop's gen and Phi
    def phi_mat(a):
        return torch.einsum('ba, ncaj -> ncbj', Phi, a)


    def real2complex(A):
        C = A.shape[1] // 2
        return A[:, :C, :, :] + 1j * A[:, C:, :, :]


    def z2h(z):
        Gz_rev = utils.transform_reverse(gen(z), mean, std)
        return real2complex(utils.vectorize(Gz_rev))


    def loss_inner(m_i, Phi, z):
        n_batch = m_i.shape[0]
        gz = z2h(z)
        std_norm = np.linalg.norm(std)
        return (norm((m_i - phi_mat(gz)).reshape(n_batch, -1), dim=1) / std_norm) ** 2


    # Setup dataloader for this M
    TestDataloader = torch.utils.data.DataLoader(
        utils.Measurement(H_all[:opt.n_test], h_all[:opt.n_test], Phi.cpu(), fixed_snr, transform=transform),
        # <-- ADD .cpu() HERE
        batch_size=opt.n_test, shuffle=False, pin_memory=False, num_workers=0
    )

    total_rate = {t: 0 for t in T_coh_list}
    batches = 0
    pbar = tqdm(total=opt.n_restart * opt.gd_step, desc=f'M={n_p} Optimizing')

    for batch, (H_batch, h_batch, y_batch) in enumerate(TestDataloader):
        H_batch, y_batch, h_batch = H_batch.to(device), y_batch.to(device), h_batch.to(device)
        cur_bch = H_batch.shape[0]

        # Optimize z (Online Phase)
        z_best = torch.randn([cur_bch, opt.latent_dim], requires_grad=False, device=device)
        for restart in range(opt.n_restart):
            z = torch.randn([cur_bch, opt.latent_dim], requires_grad=True, device=device)
            optimizer_z = optim.Adam([z], lr=opt.lr_est, betas=opt.betas_est)
            for step in range(opt.gd_step):
                optimizer_z.zero_grad()
                err_recon = loss_inner(y_batch, Phi, z)
                err_recon.backward(torch.ones_like(err_recon))
                optimizer_z.step()
                with torch.no_grad():
                    z /= z.norm(dim=1, keepdim=True)
                    z_best = z.detach().clone()
                    pbar.update(1)

        # Generator outputs the estimated angular channel
        h_est = z2h(z_best).detach()

        # Calculate EAR for each Coherence Time
        for t in T_coh_list:
            ear = utils_ear.calculate_ear(h_batch, h_est, fixed_snr, n_p, t)
            total_rate[t] += ear

        batches += 1

    pbar.close()

    # Store and print results for this M
    for t in T_coh_list:
        results_ear[t].append(total_rate[t] / batches)
        print(f"EAR (T_coh={t}): {results_ear[t][-1]:.2f} bps/Hz")

# Final summary payload to copy to plotting script
print("\n=== FINAL EAR PLOTTING DATA ===")
print(f"measurements = {measurement_list}")
for t in T_coh_list:
    print(f"ear_T{t} = {results_ear[t]}")