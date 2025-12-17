import argparse

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re

from pathlib import Path

pattern = re.compile(r"^tv_(?P<name>[^_]+)_(?P<kind>orig|adv)$")

def repl(match):
    name = match.group("name")
    kind = match.group("kind")
    return (
        f"input-{name}"
        if kind == "orig"
        else f"reconstruction-{name}"
    )

def create_ridge_plot(results):
    df_long = (
        results
        .rename(columns=lambda c: c.replace('x_', 'input-').replace('y_', 'reconstruction-'))
        .rename(columns=lambda c: re.sub(pattern, repl, c))
        .melt(var_name='source_metric', value_name='value')
    )

    df_long[['source', 'metric']] = df_long['source_metric'].str.split('-', expand=True)
    df_long = df_long.drop(columns='source_metric')

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    g = sns.FacetGrid(
        df_long,
        row="metric",
        hue="source",
        aspect=4,
        height=1.4,
        sharex=False
    )

    g.map(
        sns.histplot,
        "value",
        bins="fd",
        stat="count",
        element="step",
        fill=True,
        common_norm=False,
        alpha=0.5
    )

    g.map(plt.axhline, y=0, lw=1, clip_on=False)
    g.axes_dict["mse"].set_xscale("log")

    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    g.add_legend(title="")

    for metric, ax in g.axes_dict.items():
        ax.text(
            -0.02, 0.5,
            metric.upper(),
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
            clip_on=False
        )

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

results = df[['x_psnr', 'y_psnr', 'x_mse', 'y_mse', 'x_ssim', 'y_ssim']]
print(results.describe())

# print best and worst examples
values = df['y_psnr'].to_numpy()
idx = np.argsort(abs(values - np.median(values)))
print(f"Median: {idx[:5]}")

idx = np.argsort(values)
print(f"Lowest PSNR: {idx[:5]}")
print(f"Highest PSNR: {idx[-5:]}")

# create plots
create_ridge_plot(df[['x_psnr', 'y_psnr', 'x_mse', 'y_mse', 'x_ssim', 'y_ssim']])
plt.savefig(f'plots/{args.model}-{args.coil}-{args.organ}-ridge.pdf')
plt.close()

create_ridge_plot(df[['tv_psnr_orig', 'tv_psnr_adv', 'tv_mse_orig', 'tv_mse_adv', 'tv_ssim_orig', 'tv_ssim_adv']])
plt.savefig(f'plots/{args.model}-{args.coil}-{args.organ}-ridge-tv.pdf')
plt.close()

create_plot(df, 'psnr', 'PSNR')
plt.savefig(f'plots/{args.model}-{args.coil}-{args.organ}-psnr.pdf')
plt.close()

create_plot(df, 'mse', 'NRMSE')
plt.savefig(f'plots/{args.model}-{args.coil}-{args.organ}-nrmse.pdf')
plt.close()

create_plot(df, 'ssim', 'SSIM')
plt.savefig(f'plots/{args.model}-{args.coil}-{args.organ}-ssim.pdf')
plt.close()

create_tv_plot(df, 'psnr', 'PSNR')
plt.savefig(f'plots/{args.model}-{args.coil}-{args.organ}-psnr-tv.pdf')
plt.close()

create_tv_plot(df, 'mse', 'NRMSE')
plt.savefig(f'plots/{args.model}-{args.coil}-{args.organ}-nrmse-tv.pdf')
plt.close()

create_tv_plot(df, 'ssim', 'SSIM')
plt.savefig(f'plots/{args.model}-{args.coil}-{args.organ}-ssim-tv.pdf')
plt.close()
