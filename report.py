import argparse

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from pathlib import Path

from tabulate import tabulate

# parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('-out', type=str, default='./out', help='output directory')
parser.add_argument('-model', type=str, default='unet', choices=['unet', 'varnet'], help='model to use for reconstruction')
parser.add_argument('-organ', type=str, default='knee', choices=['knee', 'brain'])
parser.add_argument('-coil', type=str, default='sc', choices=['sc', 'mc'], help='single-coil (sc) or multi-coil (mc)')
parser.add_argument('-shape', type=str, default='line', choices=['line', 'square'], help='artefact type')

args = parser.parse_args()

# compute metrics
summaries = {
    'total_residual': [],
    'masked_residual': [],
    'unmasked_residual': [],
    'total_residual_tgt': [],
    'masked_residual_tgt': [],
    'unmasked_residual_tgt': [],
}
csvpath = Path(args.out) / args.model / f"{args.coil}_{args.organ}" / args.shape / "scores.csv"
if csvpath.exists():
    data = pd.read_csv(csvpath)

    for k in summaries.keys():
        err = 1.96*data[k].std()/np.sqrt(len(data))
        summaries[k].append(f"{data[k].mean():.2f} ({err:.2f})")

    df = pd.DataFrame(summaries)
    print(tabulate(df, headers='keys', tablefmt='psql'))

    zs = data['perturbation']
    mu, err = zs.mean(), 1.96*zs.std() / np.sqrt(len(zs))
    print(f"Perturbation: {mu:.2f} +- {err:.2f}")

    xs = data['unmasked_residual']
    ys = data['masked_residual_tgt']
    plt.scatter(xs, ys)

    plt.title(f"{args.organ} ({args.coil}) - {args.shape}")
    plt.xlabel('ground truth')
    plt.ylabel('target')
    plt.show()
