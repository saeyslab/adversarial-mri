import argparse

import torch

import toml

import csv

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

from pathlib import Path

from data import MRIDataset

import utils 
from models.unet import UNet
from models.varnet import VarNet

from fgsm import TargetedFGSM

from skimage.metrics import peak_signal_noise_ratio as psnr

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
csvpath = csvdir / "scores.csv"

figpath = outpath / args.shape / "figures"
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
    writer = csv.DictWriter(csvfile, fieldnames=[
        'perturbation',
        'total_residual', 'masked_residual', 'unmasked_residual',
        'total_residual_tgt', 'masked_residual_tgt', 'unmasked_residual_tgt',
        'x_psnr', 'y_psnr'
    ])
    writer.writeheader()

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

        # compute metrics
        x0 = utils.normalize(x0)
        y0 = utils.normalize(y0)
        x_adv = utils.normalize(x_adv)
        y_adv = utils.normalize(y_adv)

        u = np.sqrt(np.square(x0 - x_adv.cpu().detach().numpy()).sum())
        
        v0 = torch.sqrt(torch.square(y0 - y_adv).sum())
        v1 = torch.sqrt((torch.square(y0 - y_adv) * m).sum())
        v2 = torch.sqrt((torch.square(y0 - y_adv) * (1 - m)).sum())
        
        w0 = torch.sqrt(torch.square(y_tgt - y_adv).sum())
        w1 = torch.sqrt((torch.square(y_tgt - y_adv) * m).sum())
        w2 = torch.sqrt((torch.square(y_tgt - y_adv) * (1 - m)).sum())
        writer.writerow({
            'perturbation': u,

            'total_residual': v0.item(),
            'masked_residual': v1.item(),
            'unmasked_residual': v2.item(),

            'total_residual_tgt': w0.item(),
            'masked_residual_tgt': w1.item(),
            'unmasked_residual_tgt': w2.item(),

            'x_psnr': psnr(x0, x_adv.cpu().detach().numpy()),
            'y_psnr': psnr(y0.cpu().detach().numpy(), y_adv.cpu().detach().numpy())
        })
        csvfile.flush()

        # plot result
        fig, axes = plt.subplots(2, 2)
        axes[0,0].set_title(r"$x$")
        axes[0,0].imshow(x0.squeeze(), cmap='gray')
        axes[0,0].set_axis_off()

        axes[1,0].set_title(r"$F(x)$")
        axes[1,0].imshow(y0.cpu().detach().numpy().squeeze(), cmap='gray')
        axes[1,0].set_axis_off()

        axes[0,1].set_title(r"$\tilde x$")
        axes[0,1].imshow(x_adv.cpu().detach().numpy().squeeze(), cmap='gray')
        axes[0,1].set_axis_off()

        axes[1,1].set_title(r"$F(\tilde x)$")
        axes[1,1].imshow(y_adv.cpu().detach().numpy().squeeze(), cmap='gray')
        axes[1,1].set_axis_off()

        plt.tight_layout()
        plt.savefig(figpath / f"fig{idx:04d}.pdf")
        plt.close()
