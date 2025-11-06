import argparse

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path

def create_plot(df, metric, label, mask=False):
    if mask:
        plt.boxplot([df[f'x_{metric}_mask'], df[f'y_{metric}_mask']], tick_labels=['input', 'reconstructions'])
    else:
        plt.boxplot([df[f'x_{metric}'], df[f'y_{metric}']], tick_labels=['input', 'reconstructions'])
    plt.xlabel('data')
    plt.ylabel(label)

def create_tv_plot(df, metric, label):
    plt.boxplot([df[f'tv_{metric}_orig'], df[f'tv_{metric}_adv']], tick_labels=['original', 'perturbed'])
    plt.xlabel('data')
    plt.ylabel(label)

# parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('-out', type=str, default='./out', help='output directory')
parser.add_argument('-model', type=str, default='unet', choices=['unet', 'varnet'], help='model to use for reconstruction')
parser.add_argument('-organ', type=str, default='knee', choices=['knee', 'brain'])
parser.add_argument('-coil', type=str, default='sc', choices=['sc', 'mc'], help='single-coil (sc) or multi-coil (mc)')
parser.add_argument('-shape', type=str, default='line', choices=['line', 'square'], help='artefact type')

args = parser.parse_args()

# load data
outpath = Path(args.out) / args.model / f"{args.coil}_{args.organ}" / args.shape / "scores.csv"
assert outpath.exists(), f'Path does not exist: {outpath}'

df = pd.read_csv(outpath)

# print best and worst examples
values = df['y_psnr'].to_numpy()
idx = np.argsort(abs(values - np.median(values)))
print(f"Median: {idx[:5]}")

idx = np.argsort(values)
print(f"Lowest PSNR: {idx[:5]}")
print(f"Highest PSNR: {idx[-5:]}")

# create plots
create_plot(df, 'psnr', 'PSNR')
plt.savefig(f'plots/{args.model}-{args.coil}-{args.organ}-psnr.pdf')

create_plot(df, 'mse', 'NRMSE')
plt.savefig(f'plots/{args.model}-{args.coil}-{args.organ}-nrmse.pdf')

create_plot(df, 'ssim', 'SSIM')
plt.savefig(f'plots/{args.model}-{args.coil}-{args.organ}-ssim.pdf')

create_tv_plot(df, 'psnr', 'PSNR')
plt.savefig(f'plots/{args.model}-{args.coil}-{args.organ}-psnr-tv.pdf')

create_tv_plot(df, 'mse', 'NRMSE')
plt.savefig(f'plots/{args.model}-{args.coil}-{args.organ}-nrmse-tv.pdf')

create_tv_plot(df, 'ssim', 'SSIM')
plt.savefig(f'plots/{args.model}-{args.coil}-{args.organ}-ssim-tv.pdf')
