# Deep Causal Clustering with Counterfactual Regularization

This repository contains the full DCCR model and the AD1 configuration used in the accompanying manuscript. The default command trains one model from scratch with seed 342. The five seeds used for the reported AD1 mean can also be run together.

DCCR encodes a graph of subjects with a variational graph autoencoder and partitions the deterministic encoder means by spectral clustering. After warm-up, it fits a directed acyclic graph over latent coordinates and propagates interventions while holding each subject's structural residuals fixed. Accepted positive counterfactuals regularize the local neighborhood distribution. Accepted negative counterfactuals enter a ranking loss that favors observed same-cluster neighbors.

## Run AD1

The archived Windows CPU environment used Python 3.8.20. From the repository root, install the pinned dependencies in your Python environment and run:

```console
python -m pip install -r requirements.txt
python run_ad1.py
```

If the pinned packages are already installed, only the second command is needed. By default the loader reads the following MATLAB arrays from **`D:/work/experiments/DCGC_CF/data`**:

| File | MATLAB variable | Contents |
| --- | --- | --- |
| `AD_connectivity.mat` | `connectivities` | 96 subjects × 888 functional-connectivity features |
| `AD_labels.mat` | `VarName1` | One reference label per subject, in the same order as the feature rows |

The input MAT files are supplied separately. On another machine, pass their containing directory:

```console
python run_ad1.py --data-dir "/path/to/AD1"
```

To reproduce the five-run AD1 experiment:

```console
python run_ad1.py --seeds 42 142 242 342 442
```

Each invocation creates a new directory under `outputs/` and prints its full path. Use `--output-dir PATH` to change the parent directory. The run configuration records the hashes and MATLAB variable names of the two input files, package versions, and resolved model settings.

## Model and evaluation settings

| Setting | AD1 main experiment |
| --- | --- |
| Structural, positive and negative loss weights | 0.01, 0.25, 10.0 |
| Training budget | 1,000 updates, including 100 VGAE warm-up updates |
| Encoder dimensions | 64 hidden units, 16 latent coordinates |
| Optimization | Adam, learning rate 0.01, dropout 0.01 |
| Graph and partition | Symmetric cosine 8-nearest-neighbor input graph, 10-nearest-neighbor spectral clustering, 4 clusters |

The complete settings, including the DAG optimizer and intervention filters, are in [configs/ad1.json](configs/ad1.json). This is a transductive experiment. Feature standardization uses all 96 subjects. The model receives features and the known cluster count. A separate evaluator uses the reference labels to score each partition. 


## Check a run

Compare the generated `per_seed.csv` with [reference/ad1_metrics.csv](reference/ad1_metrics.csv). The reference file contains the original five runs and both selection rules. For the default seed 42, the archived values are:

| Epoch | ACC (%) | NMI (%) | ARI (%) |
|  ---: | ---: | ---: | ---: |
|  ACC | 374 | 89.58 | 72.39 | 73.17 |


Across the five original runs, the ACC results were **89.38 ± 1.36% ACC**, **71.60 ± 2.59% NMI**, and **72.70 ± 3.53% ARI** (mean ± sample standard deviation). CSV scores are fractions. Operating systems and numerical libraries can affect the optimization trajectory.



## Source

The training code is organized into [the graph encoder and training loop](dccr/model.py), [structural estimation and intervention propagation](dccr/scm.py), and [counterfactual selection and losses](dccr/counterfactual.py). [Data loading](dccr/data.py)