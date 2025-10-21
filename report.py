import argparse

import torch

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import toml

from data import Sample, MRIDataset

from skimage.metrics import peak_signal_noise_ratio as psnr

from tqdm import tqdm

from utils import zero_fill

from pathlib import Path

from models.unet import UNet
from models.varnet import VarNet

from tabulate import tabulate

# parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('data', type=str, help='path to the fastMRI data set')
parser.add_argument('-out', type=str, default='./out', help='output directory')
parser.add_argument('-model', type=str, default='unet', choices=['unet', 'varnet'], help='model to use for reconstruction')
parser.add_argument('-organ', type=str, default='knee', choices=['knee', 'brain'])
parser.add_argument('-coil', type=str, default='sc', choices=['sc', 'mc'], help='single-coil (sc) or multi-coil (mc)')
parser.add_argument('-shape', type=str, default='line', choices=['line', 'square'], help='artefact type')

args = parser.parse_args()

# get device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"[*] Device: {device}")

# load config file
config = toml.load('config.toml')

# load dataset
datapath = Path(args.data)
datasplit = "multicoil_test" if args.coil == "mc" else "singlecoil_test"
dataset = MRIDataset(datapath / f"{args.organ}/{datasplit}")
print(f"Loaded {len(dataset)} samples.")

# load model
model_modules = {
    "unet": UNet,
    "varnet": VarNet
}
outpath = Path(args.out) / args.model / f"{args.coil}_{args.organ}"
assert outpath.exists(), f'Directory does not exist: {outpath}'
weightpath = outpath / f"{args.model}.pt"
model = model_modules[args.model](args.organ, args.coil, weightpath, config, device=device)

# compute metrics
summaries = {
    'x_psnr': [],
    'y_psnr': [],
    'x_mse': [],
    'y_mse': []
}
path = outpath / args.shape
if path.exists():
    with PdfPages(path / 'plots.pdf') as pdf:
        for sample in tqdm(dataset):
            fname = sample.metadata['fname'].split('/')[-1]
            result = path / f"{fname}.npy"
            if result.exists():
                # choose slice
                idx = sample.num_slices // 2
                sample = sample.at_slice(sample.num_slices // 2)

                # load perturbation
                delta = np.load(result)
                adv_sample = Sample.from_numpy(sample.kspace + delta, sample.mask, sample.metadata)

                # get inputs
                orig_image = zero_fill(sample).squeeze()
                adv_image = zero_fill(adv_sample).squeeze()
                x_mse = np.sqrt(np.square(orig_image - adv_image).sum())

                # reconstruct outputs
                with torch.no_grad():
                    orig_output = model(sample).cpu().detach().numpy().squeeze()
                    adv_output = model(adv_sample).cpu().detach().numpy().squeeze()
                y_mse = np.sqrt(np.square(orig_output - adv_output).sum())
                
                # plot results
                fig, axes = plt.subplots(2, 2)
                axes[0, 0].imshow(orig_image, cmap='gray')
                axes[0, 0].axis('off')
                axes[0, 1].imshow(adv_image, cmap='gray')
                axes[0, 1].axis('off')

                axes[1, 0].imshow(orig_output, cmap='gray')
                axes[1, 0].axis('off')
                axes[1, 1].imshow(adv_output, cmap='gray')
                axes[1, 1].axis('off')

                plt.tight_layout()
                pdf.savefig()
                plt.close()

                # save metrics
                summaries['x_mse'].append(x_mse)
                summaries['y_mse'].append(y_mse)
                summaries['x_psnr'].append(psnr(orig_image, adv_image))
                summaries['y_psnr'].append(psnr(orig_output, adv_output))
    
    for k in summaries.keys():
        summaries[k] = [np.mean(summaries[k]), 1.96 * np.std(summaries[k]) / np.sqrt(len(summaries[k]))]

    df = pd.DataFrame(summaries)
    df.index = ["mean", "error"]
    print(tabulate(df, headers='keys', tablefmt='pretty'))
else:
    raise FileNotFoundError(path)
