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
    'x_psnr': [],
    'y_psnr': []
}
csvpath = Path(args.out) / args.model / f"{args.coil}_{args.organ}" / args.shape / "scores.csv"
if csvpath.exists():
    data = pd.read_csv(csvpath)

    for k in summaries.keys():
        err = 1.96*data[k].std()/np.sqrt(len(data))
        summaries[k].append(f"{data[k].mean():.2f} ({err:.2f})")

    df = pd.DataFrame(summaries)
    print(tabulate(df, headers='keys', tablefmt='psql'))

    x_psnr = data['x_psnr']
    y_psnr = data['y_psnr']

    plt.title(f"{args.model}: {args.organ} ({args.coil}) - {args.shape}")
    plt.boxplot([x_psnr, y_psnr], tick_labels=['input', 'output'])
    plt.xlabel('metric')
    plt.ylabel('PSNR')
    plt.show()

    ts = x_psnr / y_psnr
    idx = np.argsort(ts).to_list()
    print(f'Top scoring  : {idx[:10]}')
    print(f'Worst scoring: {idx[-10:]}')
else:
    raise FileNotFoundError(csvpath)
