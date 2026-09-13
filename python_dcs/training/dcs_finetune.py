import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

"""
dcs_finetune_stage2.py
=========================================
"The Scalpel": Precision Fine-Tuning Pass.
Uses 99% L1 Loss, extremely low learning rates, and higher inner-loop steps
to push the NMSE below the physical -16.7 dB barrier without oscillating.
"""

import numpy as np
import os
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import utils
from torch import tanh
from torch.linalg import norm
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Generator

def main():
    # --- CONFIGURATION ---
    # 1. Chain from your best Phase 1 fine-tuning checkpoint
    CKPT_PATH = "ckpt/dcs_r32t256k8_finetuned/finetuned_e15_l1.pth.tar"
    SAVE_DIR = "ckpt/dcs_r32t256k8_finetuned_stage2"
    os.makedirs(SAVE_DIR, exist_ok=True)

    BATCH_SIZE = 50
    N_EPOCHS = 15  # Only 10-15 epochs needed for this final sharpening

    # 2. "The Scalpel" Learning Rates & Steps
    LR_GEN = 5e-6  # Slower outer-loop updates to protect learned paths
    LR_EST = 0.01  # Smaller inner-loop steps to prevent z from bouncing
    GD_STEP = 150  # More steps to allow the slower z to reach the absolute bottom

    N_P = 100
    LATENT_DIM = 100
    IMG_SIZE = (16, 32, 256)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Starting Stage 2 Precision Fine-Tuning on {device}...")

    # --- 1. LOAD PRE-TRAINED MODEL ---
    gen = Generator(IMG_SIZE, LATENT_DIM).to(device)
    print(f"Loading weights from {CKPT_PATH}")
    checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    gen.load_state_dict(checkpoint['generator_state_dict'])

    opt_gen = optim.Adam(gen.parameters(), lr=LR_GEN, betas=(0.9, 0.99))

    # --- 2. DATA LOADING & NORMALIZATION ---
    H_all, h_all, _, _ = torch.load('./data/channel-r32t256k8-n5000.pt', weights_only=False)

    std_tensor = H_all.std(dim=(0, 2, 3), keepdim=True)
    mean_tensor = H_all.mean(dim=(0, 2, 3), keepdim=True)
    safe_floor = max(1e-6, 1e-3 * std_tensor.mean().item())
    std_tensor = torch.clamp(std_tensor, min=safe_floor)

    mean_seq = mean_tensor.squeeze().tolist()
    std_seq = std_tensor.squeeze().tolist()
    std_norm = np.linalg.norm(std_seq)

    transform = transforms.Compose([transforms.Normalize(mean_seq, std_seq)])
    mean = torch.tensor(mean_seq, device=device)
    std = torch.tensor(std_seq, device=device)

    dataloader = DataLoader(
        utils.ChannelDataset(H_all, h_all, transform=transform),
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0
    )

    Phi = utils.get_measurement_matrix(256, 32, N_P, 4, 4, 4, 417).to(device).to(torch.complex64)

    # --- 3. HELPER FUNCTIONS ---
    def phi_mat(a):
        return torch.einsum('ba, ncaj -> ncbj', Phi, a)

    def real2complex(A):
        C = A.shape[1] // 2
        return A[:, :C, :, :] + 1j * A[:, C:, :, :]

    def z2h(z, reverse=True):
        Gz = gen(z)
        Gz_rev = utils.transform_reverse(Gz, mean, std)
        gz = utils.vectorize(Gz_rev) if reverse else utils.vectorize(Gz)
        return real2complex(gz)

    # --- 4. THE 99% L1 HYBRID LOSS FUNCTIONS ---
    def loss_inner(m_i, Phi, z):
        n_batch = m_i.shape[0]
        gz = z2h(z, reverse=True)
        diff = (m_i - phi_mat(gz)).reshape(n_batch, -1)

        err_l2 = (norm(diff, dim=1) / std_norm) ** 2
        err_l1 = norm(diff, ord=1, dim=1) / std_norm

        # 99% L1 to ruthlessly penalize microscopic errors
        return 0.01 * err_l2 + 0.99 * err_l1

    def nmse_compute(z, h_bch):
        n_batch = z.shape[0]
        gz = z2h(z, reverse=True)
        diff = (h_bch - gz).reshape(n_batch, -1)

        # Original L2 NMSE (for terminal logging)
        err_l2 = norm(diff, dim=1) ** 2
        h_norm_l2 = norm(h_bch.reshape(n_batch, -1), dim=1) ** 2
        nmse_l2 = err_l2 / h_norm_l2

        # L1 Error (for updating the Generator weights)
        err_l1 = norm(diff, ord=1, dim=1)
        h_norm_l1 = norm(h_bch.reshape(n_batch, -1), ord=1, dim=1)
        nmse_l1 = err_l1 / h_norm_l1

        # 99% L1 to aggressively force high-precision rank updates
        hybrid_loss = 0.01 * nmse_l2 + 0.99 * nmse_l1
        return hybrid_loss, nmse_l2

    def loss_outer(hybrid_err_recon, Phi, x1, x2, x3):
        loss_G = hybrid_err_recon.mean()
        n_batch = hybrid_err_recon.shape[0]

        def get_rip_loss(img1, img2):
            m1 = (phi_mat(img1)).reshape(n_batch, -1)
            m2 = (phi_mat(img2)).reshape(n_batch, -1)
            img_diff_norm = norm((img1 - img2).reshape(n_batch, -1), dim=1)
            m_diff_norm = norm(m1 - m2, dim=1)
            return (img_diff_norm / m_diff_norm - np.sqrt(Phi.shape[1] / Phi.shape[0])) ** 2 * (Phi.shape[0] / Phi.shape[1])

        r1, r2, r3 = get_rip_loss(x1, x2), get_rip_loss(x1, x3), get_rip_loss(x2, x3)
        loss_F = ((r1 + r2 + r3) / 3.0).mean()
        return loss_G, loss_F

    def sample_and_opt(m, h):
        z = torch.randn([BATCH_SIZE, LATENT_DIM], requires_grad=True, device=device)
        opt_z = optim.Adam([z], lr=LR_EST, betas=(0.9, 0.99))
        gen_img_initial = z2h(z.clone())

        for _ in range(GD_STEP):
            opt_z.zero_grad()
            err_recon = loss_inner(m, Phi, z)
            err_recon.backward(torch.ones_like(err_recon))
            opt_z.step()
            with torch.no_grad():
                z /= z.norm(dim=1, keepdim=True)

        gen_img_final = z2h(z.clone())
        hybrid_loss, nmse_l2 = nmse_compute(z, h)
        return hybrid_loss, nmse_l2, gen_img_initial, gen_img_final

    # --- 5. FINE-TUNING LOOP ---
    for epoch in range(1, N_EPOCHS + 1):
        pbar = tqdm(total=H_all.shape[0] // BATCH_SIZE, desc=f'Stage 2 Epoch [{epoch}/{N_EPOCHS}]')

        for batch, (H, h) in enumerate(dataloader):
            H, h = H.to(device), h.to(device)
            m = phi_mat(h)

            hybrid_loss, nmse_l2, gen_img_initial, gen_img_final = sample_and_opt(m, h)
            loss_G, loss_F = loss_outer(hybrid_loss, Phi, h, gen_img_initial, gen_img_final)
            loss_meta = loss_G + loss_F

            opt_gen.zero_grad()
            loss_meta.backward()
            opt_gen.step()

            pbar.update(1)
            pbar.set_postfix({'Loss': loss_meta.item(), 'True NMSE (dB)': 10 * np.log10(nmse_l2.mean().item())})

        pbar.close()

        # Save Checkpoint
        if epoch % 5 == 0 or epoch == N_EPOCHS:
            save_name = f"{SAVE_DIR}/finetuned_stage2_e{epoch}.pth.tar"
            torch.save({
                'generator_state_dict': gen.state_dict(),
                'mean': mean_seq,
                'std': std_seq
            }, save_name)
            print(f"Saved surgical weights to: {save_name}")

if __name__ == "__main__":
    main()