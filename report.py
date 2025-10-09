import argparse

import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path

from tabulate import tabulate

# parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('-out', type=str, default='./out', help='output directory')

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
labels = []
scores = []
for organ in ['knee', 'brain']:
    for coil in ['sc', 'mc']:
        for shape in ['line', 'square']:
            csvpath = Path(args.out) / f"{coil}_{organ}" / shape / "scores.csv"
            if csvpath.exists():
                data = pd.read_csv(csvpath)
                scores.append(data)

                for k in summaries.keys():
                    summaries[k].append(f"{data[k].mean():.2f} ({data[k].std():.2f})")

                label = f"{organ} ({coil}) - {shape}"
                labels.append(label)

df = pd.DataFrame(summaries)
print(tabulate(df, headers='keys', tablefmt='psql'))

for i in range(len(scores)):
    xs = scores[i]['masked_residual'] / scores[i]['total_residual']
    ys = scores[i]['masked_residual_tgt'] / scores[i]['total_residual_tgt']

    plt.scatter(xs, ys, label=labels[i])

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('ground truth')
plt.ylabel('target')
plt.legend()
plt.show()
