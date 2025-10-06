import argparse

import random

import torch

import toml

import csv

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
parser.add_argument('-eps', type=float, default=.01, help='maximum perturbation size')
parser.add_argument('-step', type=float, default=.001, help='attack step size')
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
outpath = Path(args.out) / f"{args.coil}_{args.organ}"
outpath.mkdir(exist_ok=True, parents=True)
weightpath = outpath / f"{args.model}.pt"

csvpath = outpath / "scores.csv"

figpath = outpath / "figures"
figpath.mkdir(exist_ok=True)

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
with open(csvpath, "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['index', 'diff_x', 'diff_y0', 'diff_yt'])
    writer.writeheader()

    for idx, sample in enumerate(tqdm(dataset)):
        # zero-fill the entire sample
        zf = utils.zero_fill(sample)

        # choose a slice
        slice_arr = zf[random.randint(0, sample.num_slices - 1)]
        x0 = torch.from_numpy(slice_arr).float().unsqueeze(0).to(device)

        # reconstruct
        with torch.no_grad():
            y0 = model(x0)
        
        # construct mask
        mask_params = mask_drawings[args.shape]
        mask = utils.make_xdet_cv_like(y0,
                                kind=args.shape,
                                size=mask_params['size'], thickness=mask_params['thickness'],
                                value=1.0).to(device, dtype=x0.dtype)
        
        # run attack
        x_adv, y_adv, y0, y_tgt, m = attacker(x0, mask=mask, w_in=1)
        
        y_rng = (y0.max() - y0.min()).item()
        loss1 = abs(y0 - y_adv).max().item() / y_rng
        loss2 = abs(y_adv - y_tgt).max().item() / y_rng
        writer.writerow({
            'index': idx,
            'diff_y0': loss1,
            'diff_yt': loss2
        })

        # plot result
        fig, axes = plt.subplots(2, 2)
        axes[0,0].set_title(r"$x$")
        axes[0,0].imshow(utils.normalize(x0).cpu().detach().numpy().squeeze(), cmap='gray')
        axes[0,0].set_axis_off()

        axes[1,0].set_title(r"$F(x)$")
        axes[1,0].imshow(utils.normalize(y0).cpu().detach().numpy().squeeze(), cmap='gray')
        axes[1,0].set_axis_off()

        axes[0,1].set_title(r"$\tilde x$")
        axes[0,1].imshow(utils.normalize(x_adv).cpu().detach().numpy().squeeze(), cmap='gray')
        axes[0,1].set_axis_off()

        axes[1,1].set_title(r"$F(\tilde x)$")
        axes[1,1].imshow(utils.normalize(y_adv).cpu().detach().numpy().squeeze(), cmap='gray')
        axes[1,1].set_axis_off()

        plt.tight_layout()
        plt.savefig(figpath / f"fig{idx:04d}.pdf")
