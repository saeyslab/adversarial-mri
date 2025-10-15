import argparse

import torch

import toml

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

from pathlib import Path

from data import MRIDataset

import utils 
from models.unet import UNet
from models.varnet import VarNet

from fgsm import TargetedFGSM

# parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('data', type=str, help='path to the fastMRI data set')
parser.add_argument('-out', type=str, default='./out', help='output directory')
parser.add_argument('-model', type=str, default='unet', choices=['unet', 'varnet'], help='model to use for reconstruction')
parser.add_argument('-iterations', type=int, default=150, help='number of iterations of the attack')
parser.add_argument('-eps', type=float, default=1e-6, help='maximum perturbation size')
parser.add_argument('-step', type=float, default=1e-7, help='attack step size')
parser.add_argument('-organ', type=str, default='knee', choices=['knee', 'brain'])
parser.add_argument('-coil', type=str, default='sc', choices=['sc', 'mc'], help='single-coil (sc) or multi-coil (mc)')
parser.add_argument('-shape', type=str, default='line', choices=['line', 'square'], help='artefact type')

args = parser.parse_args()

# verify arguments
datapath = Path(args.data)
assert datapath.exists(), f'Directory does not exist: {datapath}'

# get device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"[*] Device: {device}")

# load config file
config = toml.load('config.toml')

# create directories
outpath = Path(args.out) / args.model / f"{args.coil}_{args.organ}"
outpath.mkdir(exist_ok=True, parents=True)
weightpath = outpath / f"{args.model}.pt"

csvdir = outpath / args.shape
csvdir.mkdir(exist_ok=True)

# load model
model_modules = {
    "unet": UNet,
    "varnet": VarNet
}
model = model_modules[args.model](args.organ, args.coil, weightpath, config, device=device)

# load dataset
datasplit = "multicoil_test" if args.coil == "mc" else "singlecoil_test"
dataset = MRIDataset(datapath / f"{args.organ}/{datasplit}")
print(f"Loaded {len(dataset)} samples.")

# define masks
mask_drawings = {
    "square": {
        "size": 50,
        "thickness": -1
    },
    "line": {
        "size": 60,
        "thickness": 4
    }
}

# run attack
attacker = TargetedFGSM(
    model,
    eps=args.eps,
    step_size=args.step,
    n_iter=args.iterations
)
for idx, sample in enumerate(tqdm(dataset)):
    # choose slice
    sample = sample.at_slice(sample.num_slices // 2)

    # original reconstruction
    y0 = model(sample)
    x0 = utils.zero_fill(sample)

    # construct mask
    mask_params = mask_drawings[args.shape]
    mask = utils.make_xdet_cv_like(y0,
                            kind=args.shape,
                            size=mask_params['size'], thickness=mask_params['thickness'],
                            value=1.0).to(device, dtype=y0.dtype)
    
    # run attack
    x_adv, y_adv, y_tgt, m = attacker(sample, mask=mask, w_in=1)

    # save result
    delta = x_adv.kspace.cpu().detach().numpy() - sample.kspace
    part = sample.metadata['fname'].split('/')[-1]
    np.save(csvdir / f"{part}.npy", delta)
