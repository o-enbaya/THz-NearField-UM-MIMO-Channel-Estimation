# Deep Compressed Sensing (DCS) for THz UM-MIMO Near-Field Channel Estimation

This directory contains the PyTorch implementation of the Deep Compressed Sensing (DCS) estimator used in the thesis (based on the code of Lin, Erdem and Alouini, 2025). The unchanged scripts that produced the Deep CS results in the report are in [`../thesis_figure_scripts/python/`](../thesis_figure_scripts/README.md).
> **Thesis Reference**: Chapter 3 (Sections 3.6–3.8, Algorithms 1 & 2) and Chapter 4 (Sections 4.3–4.4).

---

## 1. Environment & Setup

### Option A: Using Conda (Recommended)
Create the official conda environment using `environment.yml`:
```bash
# Automated creation (creates the environment in ./env):
chmod +x create-conda-env.sh
./create-conda-env.sh
conda activate ./env

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
│   ├── dcs_finetune.py              # Later fine-tuning experiment (not used in the report)
│   └── mat2pt.py                    # Converts MATLAB .mat channel arrays into PyTorch .pt tensors
│
├── evaluation/                      # Evaluation scripts (see the main README, Section 4, for which ones match the report)
│   ├── eval_fig4_2_nmse_vs_snr.py   # Later NMSE vs SNR version (fine-tuned checkpoint)
│   ├── eval_fig4_3_ear_vs_meas.py   # Earlier EAR version (e130 checkpoints)
│   ├── eval_fig4_4_nmse_vs_rho.py   # Earlier NMSE vs compression ratio version (e130 checkpoint)
│   ├── eval_fig4_5_ber_vs_snr.py    # Later exploratory BER version (QPSK); Figure 4.5 used 16-QAM
│   ├── eval_fig4_6_distance_sweep.py# Exploratory Deep CS distance sweep (not in Figure 4.6)
│   ├── eval_fig4_7_runtime.py       # Deep CS NMSE of Figure 4.2 and runtime of Figure 4.7(b)
│   ├── plot_dcsEAR.py               # Deep CS EAR of Figure 4.3
│   └── eval_runtime.py              # Prints Table 4.3 FLOPs and the measured runtimes
│
├── checkpoints/
│   └── README.md                    # Notes on the trained checkpoints (not included)
│
└── data/                            # Result CSV files (some from earlier runs)
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
* **Offline MAML Training**: Adam optimizer, $\text{lr} = 0.0002$, $\beta = (0.9, 0.99)$, batch size 50 (the script default is 100), 300 epochs, 10 inner steps, with an auxiliary RIP distance-preservation loss. The training measurements are noise-free.

---

## 4. Running Thesis Evaluations

Run the scripts from this folder (`python_dcs/`), because they read `./data/` and `./ckpt/` relative to the current folder. The checkpoint used in the report is `ckpt/dcs_r32t256k8/e300b50gd10np100dl100.pth.tar`.
```bash
# Deep CS NMSE of Figure 4.2 and runtime of Figure 4.7(b):
python evaluation/eval_fig4_7_runtime.py

# Deep CS EAR of Figure 4.3:
python evaluation/plot_dcsEAR.py

# Table 4.3 FLOPs and measured runtimes:
python evaluation/eval_runtime.py
```
For Figures 4.4 and 4.5, use `../thesis_figure_scripts/python/nmseVSrho gemini.py` and `../thesis_figure_scripts/python/NMSEandBER.py` (copy them into this folder first, since they import `model` and `utils`).
