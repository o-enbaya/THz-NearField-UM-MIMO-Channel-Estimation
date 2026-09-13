# Deep Compressed Sensing (DCS) Trained Model Checkpoints

This folder hosts pre-trained PyTorch weights for the DCS-MAML generator.
Because trained model weights exceed GitHub's 100 MB file limit (~560 MB - 726 MB),
they are excluded from direct Git tracking via `.gitignore`.

### Default Checkpoint Path:
`python_dcs/checkpoints/dcs_r32t256k8/e300b50gd10np100dl100.pth.tar`
or
`python_dcs/checkpoints/dcs_stage3_physics/finetuned_stage3_physics_e5.pth.tar`

To use pre-trained weights:
Copy `e0b100gd100np100dl256.pth.tar` or fine-tuned checkpoints into this directory.
