import argparse

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from pathlib import Path

from tabulate import tabulate

from scipy.spatial import ConvexHull

from utils import normalize

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

    xs = normalize(data['total_residual'])
    ys = normalize(data['total_residual_tgt'])
    plt.scatter(xs, ys)

    plt.title(f"{args.organ} ({args.coil}) - {args.shape}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('ground truth')
    plt.ylabel('target')
    plt.show()

    points = np.zeros([xs.shape[0], 2])
    points[:, 0] = xs
    points[:, 1] = ys

    hull = ConvexHull(points)
    verts = points[hull.vertices]
    for idx, vert in zip(hull.vertices, verts):
        if vert[0] < .5 and vert[1] < .5:
            print(f"{idx:04d}: {vert}")
