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
scores = []
labels = []
summaries = {
    'data': [],
    'mean': [],
    'std': []
}
for organ in ['knee', 'brain']:
    for coil in ['sc', 'mc']:
        for shape in ['line', 'square']:
            csvpath = Path(args.out) / f"{coil}_{organ}" / shape / "scores.csv"
            if csvpath.exists():
                data = pd.read_csv(csvpath)
                mu, sigma = data['metric'].mean(), data['metric'].std()

                label = f"{organ} ({coil}) - {shape}"

                summaries['data'].append(label)
                summaries['mean'].append(mu)
                summaries['std'].append(sigma)
                scores.append(data['metric'])
                labels.append(label)

df = pd.DataFrame(summaries)
print(tabulate(df, headers='keys', tablefmt='psql'))

plt.boxplot(scores, tick_labels=labels)
plt.ylim(0, 1)
plt.show()
