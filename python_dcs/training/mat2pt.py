"
mat2pt.py
Converts MATLAB-generated near-field channel tensors (.mat) into PyTorch dataset tensors (.pt).
Computes channel mean and standard deviation for normalization.
"
import os
import sys
import argparse
import torch

# Ensure python_dcs directory is in sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    import utils
except ImportError:
    from python_dcs import utils

def main():
    parser = argparse.ArgumentParser(description=Convert MATLAB .mat channel dataset to PyTorch .pt tensor format.)
    parser.add_argument(
        --input, -i,
        type=str,
        default=data/channel-r32t256k8-n5000d1.2delta0.0005Delta0.01theta0.523599-Multipath+LoS.mat,
        help=Path to the input MATLAB .mat file
    )
    parser.add_argument(
        --output, -o,
        type=str,
        default=./data/channel-r32t256k8-n5000.pt,
        help=Path to save the output PyTorch .pt tensor
    )
    args = parser.parse_args()

    print(f[mat2pt] Loading MATLAB channel data from: {args.input})
    if not os.path.exists(args.input):
        alt_input = os.path.join(parent_dir, args.input)
        if os.path.exists(alt_input):
            args.input = alt_input
        else:
            print(f[Error] File not found: {args.input})
            sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    H_all, h_all, mean, std = utils.get_channel(args.input)
    torch.save((H_all, h_all, mean, std), args.output)
    print(f[mat2pt] Successfully processed and saved tensor to: {args.output})
    print(f Tensor shape: {H_all.shape}, Vectors: {h_all.shape})

if __name__ == __main__:
    main()
