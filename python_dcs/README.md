# Deep Compressed Sensing (DCS) for THz UM-MIMO Near-Field Channel Estimation

This directory contains the authentic PyTorch deep learning implementation of the Deep Compressed Sensing (DCS) architecture for wideband near-field THz UM-MIMO channel estimation:
> **Thesis Reference**: Chapter 3 (Sections 3.6–3.8, Algorithms 1 & 2) and Chapter 4 (Sections 4.3–4.4).

---

## 1. Environment & Setup

### Option A: Using Conda (Recommended)
Create the official conda environment using `environment.yml`:
```bash
# Automated creation:
chmod +x create-conda-env.sh
./create-conda-env.sh

# Or manual creation:
conda env create -f environment.yml
conda activate torch
```

### Option B: Using Pip
```bash
pip install -r requirements.txt
```

---

## 2. Directory Structure

```
python_dcs/
│
├── environment.yml                  # Complete conda environment definition
├── create-conda-env.sh              # Shell script for conda creation
├── requirements.txt                 # Core Python dependencies
├── model.py                         # PyTorch Generator G_theta & Discriminator architectures
├── utils.py                         # Complex data formatting, ChannelDataset & measurement matrices
├── utils_ear.py                     # Effective Achievable Rate mathematical utilities
│
├── training/
│   ├── dcs_train.py                 # Algorithm 1: Offline MAML meta-training loop (300 epochs)
│   ├── dcs_finetune.py              # Latent space fine-tuning for soft BER
│   └── mat2pt.py                    # Converts MATLAB .mat channel arrays into PyTorch .pt tensors
│
├── evaluation/                      # Thesis Chapter 4 Figure Reproduction Scripts
│   ├── eval_fig4_2_nmse_vs_snr.py   # Figure 4.2: NMSE vs Transmit SNR (-20 dB to 20 dB)
│   ├── eval_fig4_3_ear_vs_meas.py   # Figure 4.3: EAR vs Pilot Measurements (Inf & 512 coherence)
│   ├── eval_fig4_4_nmse_vs_rho.py   # Figure 4.4: NMSE vs Compression Ratio (rho = Np/Nt)
│   ├── eval_fig4_5_ber_vs_snr.py    # Figure 4.5: Calibrated QPSK Bit Error Rate (BER) vs SNR
│   ├── eval_fig4_6_distance_sweep.py# Figure 4.6: Near-field Distance Sweep (0.4m to 1.6m)
│   └── eval_fig4_7_runtime.py       # Figure 4.7: Online inference execution time profiling
│
├── checkpoints/                     # Model weights directory (gitignored for large files)
│   └── README.md                    # Instructions on placing .pth / .pth.tar weights
│
└── data/                            # Calibrated evaluation CSVs & sample datasets
    ├── Classical_Baselines_Calibrated.csv
    ├── DCS_BER_vs_SNR.csv
    ├── DCS_EAR_vs_Measurements_SNR24_dl100.csv
    ├── DCS_NMSE_vs_Rho_SNR10.csv
    ├── DCS_QPSK_Np100_BER_vs_SNR.csv
    └── dcs_runtime_per_sample_per_snr.csv
```

---

## 3. Deep Learning Architecture (Table 4.2)

* **Tensor Dimensions**: $(N, 2K, N_r, N_t) = (N, 16, 32, 256)$ real-valued representation.
* **Channel-wise Normalization**: Evaluated across dataset realizations per Equations (4.12)–(4.14).
* **Latent Space**: $z \in \mathbb{R}^{100}$ standard normal prior.
* **Online Inference**: Adam optimizer, $\text{lr} = 0.1$, $\beta = (0.9, 0.99)$, unit sphere projection.
* **Offline MAML Training**: Adam optimizer, $\text{lr} = 0.0002$, $\beta = (0.9, 0.99)$, batch size 50/100, 300 epochs with auxiliary RIP distance-preservation loss.

---

## 4. Running Thesis Evaluations

From the repository root:
```bash
# Figure 4.2 (NMSE vs. SNR):
python -m python_dcs.evaluation.eval_fig4_2_nmse_vs_snr

# Figure 4.3 (EAR vs. Measurements):
python -m python_dcs.evaluation.eval_fig4_3_ear_vs_meas

# Figure 4.4 (NMSE vs. Compression Ratio):
python -m python_dcs.evaluation.eval_fig4_4_nmse_vs_rho

# Figure 4.5 (QPSK System BER vs. SNR):
python -m python_dcs.evaluation.eval_fig4_5_ber_vs_snr

# Figure 4.6 (Distance Sweep):
python -m python_dcs.evaluation.eval_fig4_6_distance_sweep

# Figure 4.7 (Runtime Profiling):
python -m python_dcs.evaluation.eval_fig4_7_runtime
```
