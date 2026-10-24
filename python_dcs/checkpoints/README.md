# Deep Compressed Sensing (DCS) Trained Model Checkpoints

The trained generator checkpoints are not included in this repository because the files are larger than GitHub's 100 MB file limit.

### Where the scripts look for them
`training/dcs_train.py` saves checkpoints to `ckpt/dcs_r32t256k8/` inside the folder it is run from, and the evaluation scripts load them from the same place. Run the scripts from `python_dcs/`, so the path is:

`python_dcs/ckpt/dcs_r32t256k8/e300b50gd10np100dl100.pth.tar`

This is the checkpoint used for the results in the report (epoch 300, batch size 50, 10 inner gradient steps, 100 pilots, latent dimension 100). `evaluation/eval_fig4_6_distance_sweep.py` is the exception: it looks in `python_dcs/checkpoints/dcs_r32t256k8/`.
