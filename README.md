# Description

This repository contains the code to reproduce our experiments for the paper *Triggering hallucinations in model-based MRI reconstruction via adversarial perturbations* (to appear).

# Setup

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

# Data set

Our experiments make use of the [fastMRI data set](https://fastmri.med.nyu.edu/), which is publicly available.
We only evaluated our attack on the brain and knee portions of the data, but prostate and breast data should work just as well.

# CLI reference

There are three scripts in the codebase meant for external use:

* `main.py`, which computes all the adversarial perturbations
* `report.py`, which computes all the metrics
* `display.py`, which plots the metrics
* `visualize.py`, which visualizes the resulting MR images

Once the environment has been set up, experiments can be run using the `main.py` script.
This script takes a number of arguments, detailed below:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `data` | string | — | Path to the fastMRI data set. |
| `--out PATH` | string | `out` | Path to the output directory where the generated perturbations shall be stored. |
| `--model {unet,varnet}` | string | `unet` | Model to target (either UNet or E2E-VarNet). |
| `--organ {knee,brain}` | string | `knee` | Organ data to use (either knee or brain). |
| `--coil {sc,mc}` | string | `sc` | Use single-coil (`sc`) or multi-coil (`mc`) data. |
| `--shape {line,square}` | string | `line` | Shape to draw on the target images. |
| `--iterations INT` | integer | `150` | Maximum number of iterations in the attack. |
| `--eps FLOAT` | float | `1e-6` | Perturbation budget of the attack. |
| `--step FLOAT` | float | `1e-7` | Attack step size. |

Note that our attack can also use squares as the target shapes, although these results were not reported in the paper since we find them to be inferior to lines.

The `main.py` script generates the perturbations and stores them in the specified output directory.
Once this script is completed, a report can be generated using `report.py`, which takes the same arguments as `main.py`.
These reports take the form of CSV files containing all measurements of the metrics detailed in the paper.

To generate the plots we showed in the paper, run the `display.py` script after generating the report.
