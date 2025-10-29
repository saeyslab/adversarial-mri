import argparse

import torch

import numpy as np

import pandas as pd

import sigpy.mri as mr

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import toml

from data import Sample, MRIDataset

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrmse

from tqdm import tqdm

from utils import make_xdet_cv_like, zero_fill, normalize

from pathlib import Path

from models.unet import UNet
from models.varnet import VarNet

from tabulate import tabulate

def compute_metrics(x, y, x_adv, y_adv, mask=None):
    if mask is None:
        x_n, y_n = normalize(x), normalize(y)
        x_adv_n, y_adv_n = normalize(x_adv), normalize(y_adv)
    else:
        x_n, y_n = normalize(mask * x), normalize(mask * y)
        x_adv_n, y_adv_n = normalize(mask * x_adv), normalize(mask * y_adv)
    
    x_n, y_n = x_n.squeeze(), y_n.squeeze()
    x_adv_n, y_adv_n = x_adv_n.squeeze(), y_adv_n.squeeze()

    x_mse, y_mse = nrmse(x_n, x_adv_n), nrmse(y_n, y_adv_n)
    x_ssim, y_ssim = ssim(x_n, x_adv_n, data_range=1), ssim(y_n, y_adv_n, data_range=1)
    x_psnr, y_psnr = psnr(x_n, x_adv_n, data_range=1), psnr(y_n, y_adv_n, data_range=1)

    return (x_mse, x_ssim, x_psnr), (y_mse, y_ssim, y_psnr)

def compute_tv_metrics(x_tilde, y_tilde, lamda=.005):
    mps = mr.app.EspiritCalib(x_tilde).run()
    y_tv = abs(mr.app.TotalVariationRecon(x_tilde, mps, lamda).run()).real

    y_tilde_n = normalize(y_tilde).squeeze()
    y_tv_n = normalize(y_tv).squeeze()

    tv_mse = nrmse(y_tilde_n, y_tv_n)
    tv_ssim = ssim(y_tilde_n, y_tv_n, data_range=1)
    tv_psnr = psnr(y_tilde_n, y_tv_n, data_range=1)

    return tv_mse, tv_ssim, tv_psnr

def plot_dist(path, x_data, y_data, title, log=False):
    plt.scatter(x_data, y_data)
    if log:
        ax = plt.gca()
        ax.set_yscale('log')
        ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(path / f"{title}.pdf")
    plt.close()

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

# compute metrics
summaries = {
    'fname': [],
    'slice': [],
    'x_psnr': [],
    'y_psnr': [],
    'x_mse': [],
    'y_mse': [],
    'x_ssim': [],
    'y_ssim': [],
    'x_psnr_mask': [],
    'y_psnr_mask': [],
    'x_mse_mask': [],
    'y_mse_mask': [],
    'x_ssim_mask': [],
    'y_ssim_mask': [],
    'tv_psnr': [],
    'tv_mse': [],
    'tv_ssim': [],
    'loss1': [],
    'loss2': []
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

                # reconstruct outputs
                with torch.no_grad():
                    orig_output_pt = model(sample)
                    orig_output = orig_output_pt.cpu().detach().numpy().squeeze()
                    adv_output = model(adv_sample).cpu().detach().numpy().squeeze()

                # construct mask
                mask_params = mask_drawings[args.shape]
                mask = make_xdet_cv_like(orig_output_pt,
                                        kind=args.shape,
                                        size=mask_params['size'], thickness=mask_params['thickness'],
                                        value=1.0).cpu().detach().numpy()
                
                # compute loss
                y_rng = orig_output.max() - orig_output.min()
                alpha_eff = .3 * y_rng
                y_tgt = orig_output + alpha_eff * mask
                loss1 = np.square(mask * (adv_output - y_tgt)).sum() / np.sum(mask)
                loss2 = np.square((1 - mask) * (adv_output - orig_output)).sum() / np.sum(1 - mask)
                
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
                (x_mse, x_ssim, x_psnr), (y_mse, y_ssim, y_psnr) = compute_metrics(orig_image, orig_output, adv_image, adv_output)
                (x_mse_mask, x_ssim_mask, x_psnr_mask), (y_mse_mask, y_ssim_mask, y_psnr_mask) = compute_metrics(orig_image, orig_output, adv_image, adv_output, mask)
                tv_mse, tv_ssim, tv_psnr = compute_tv_metrics(adv_sample.kspace, adv_image)

                summaries['fname'].append(fname)
                summaries['slice'].append(idx)

                summaries['x_mse'].append(x_mse)
                summaries['y_mse'].append(y_mse)
                summaries['x_psnr'].append(x_psnr)
                summaries['y_psnr'].append(y_psnr)
                summaries['x_ssim'].append(x_ssim)
                summaries['y_ssim'].append(y_ssim)

                summaries['loss1'].append(loss1)
                summaries['loss2'].append(loss2)

                summaries['x_mse_mask'].append(x_mse_mask)
                summaries['y_mse_mask'].append(y_mse_mask)
                summaries['x_psnr_mask'].append(x_psnr_mask)
                summaries['y_psnr_mask'].append(y_psnr_mask)
                summaries['x_ssim_mask'].append(x_ssim_mask)
                summaries['y_ssim_mask'].append(y_ssim_mask)

                summaries['tv_mse'].append(tv_mse)
                summaries['tv_ssim'].append(tv_ssim)
                summaries['tv_psnr'].append(tv_psnr)

    df = pd.DataFrame(summaries)
    df.to_csv(path / 'scores.csv', sep=',', encoding='utf-8', index=False, header=True)
else:
    raise FileNotFoundError(path)
