# POIS-DON: POISON DATASET-DRIVEN ACTIVATION OPTIMISATION-BASED NOVELTY DETECTION

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Conference](https://img.shields.io/badge/ICIP-2026-blue)](https://2026.ieeeicip.org/)

**Authors:** Ahmed Gabr, Mahmoud Rady, Pola Qulta, Youssef Abou Eita, Youssef ElKady, Youssef Fayed, Marwan Torki  
**Affiliation:** Alexandria University, Egypt

> 🚧 **Code Release Status:** This paper has been accepted to ICIP 2026. The full codebase, including our activation optimization scripts and feature extraction pipeline, is currently being organized into a reproducible, extensible pipeline. The codebase is still a work in progress — more attacks and defenses will be added over time. The original research notebooks live in [`legacy_notebooks/`](./legacy_notebooks); the current `scripts/`, `attacks/`, `defenses/`, and related packages aim to make the experiments more reproducible and extendible. 

## Abstract

Backdoor attacks, in which a model's behavior is altered by embedding a trigger into the input to cause targeted mis-classifications, pose an increasing security threat to deep neural networks. This is particularly concerning given the widespread reliance on open-source pre-trained models to overcome data and computational constraints. 

We propose a novel, attack-agnostic, and computationally efficient method for detecting poisoned models. Our approach uses activation optimisation to generate per-class optimised logit outputs that serve as model signatures. Instead of training a meta-classifier on the signatures, our method leverages statistical features for anomaly detection. We demonstrate that our method is robust to various trigger types, attack strategies, including [BadNet](https://arxiv.org/abs/1708.06733), [Sinusoidal](https://ieeexplore.ieee.org/document/8802997), and [WaNet](https://arxiv.org/abs/2102.10369), as well as model architectures such as VGG and ResNet, without requiring retraining. We validate our approach on three popular image-classification Trojan model benchmarks: the [ULP](https://github.com/UMBCvision/Universal-Litmus-Patterns) CIFAR-10 and Tiny-ImageNet, and the ImageNet Vision Transformer-based [TAT dataset](https://github.com/vimal-isi-edu/trigs).

## Method Overview

Our proposed pipeline extracts minimization and maximization signatures for each class, which are fed back into the model to generate two logit matrices. From these, we compute the Min Votes and Max Diagonal features, concatenate them, and pass them to Pois-DON. Crucially, rather than training an external meta-classifier on the signatures, we use lightweight statistical features computed directly from the model's logits.

![Pois-DON Pipeline](./figures/pipeline_fig2.png)

## Setup

The code depends on a pinned conda environment (Python 3.10, PyTorch 2.8, etc.). Create and activate it with:

```bash
conda env create -f environment.yaml
conda activate pois-don
```

## Running the Pipeline

The pipeline proceeds in three steps: (1) download the image datasets, (2) generate poisoned or clean models, and (3) run the defense to predict which test models are clean vs. poisoned.

### Step 1: Download the image datasets

Download each dataset you plan to use with its corresponding script. Images are saved into the `storage/datasets/<dataset_name>/...` layout.

```bash
python -m image_datasets.cifar10.cifar10_download
python -m image_datasets.mnist.mnist_download
```

### Step 2: Generate poisoned or clean models

Use `scripts/run_attack.py` with an attack experiment config to train and save models. Configs live in `experiment_configs/attacks/` (e.g. `clean.yaml`, `badnet.yaml`, `sig.yaml`, `wanet_no_noise.yaml`).

```bash
python -m scripts.run_attack experiment_configs/attacks/clean.yaml
python -m scripts.run_attack experiment_configs/attacks/badnet.yaml
```

Trained models are saved under `storage/datasets/<dataset_name>/models/...`. Note that the `__main__` block in `scripts/run_attack.py` currently hardcodes a local config path — replace it with the path to your own config when running directly.

### Step 3: Run the defense

Use `scripts/run_defense.py` with a defense experiment config to predict whether each test model is clean or poisoned and export the results. Configs live in `experiment_configs/defenses/` (e.g. `poisdon.yaml`).

```bash
python -m scripts.run_defense experiment_configs/defenses/poisdon.yaml
```

Results are written under `storage/datasets/<dataset_name>/defenses/...`:
- `results.csv` — one row per test model with `model_path`, `anomaly_score`, and `is_anomaly`.
- `artifact.json` — metadata including the run timestamp, repo hash, config, and reproducibility info.

As with the attack script, the `__main__` block in `scripts/run_defense.py` hardcodes a local config path — update it to your own config path when running directly.

## Code Structure

```
.
├── scripts/                 # Entry points
│   ├── run_attack.py        #   Train & save poisoned / clean models
│   └── run_defense.py       #   Run defense, export predictions to CSV
├── image_datasets/          # Dataset configs + download scripts (cifar10, mnist)
├── attacks/                 # Attack implementations (clean_basic, sig, badnet,
│                            #   wanet_no_noise) + shared training/saving logic
├── defenses/                # Defense implementations (poisdon) + results saver
├── architectures/           # Model architectures (simple_cnn)
├── registries/              # Registry maps (architecture / attack / defense / dataset)
├── schema/                  # Dataclass schemas for attack & defense configs
├── experiment_configs/      # YAML configs for attacks and defenses
├── utils/                   # Storage layout, seeding, model hashing, repo state
└── legacy_notebooks/        # Original research notebooks used for the paper
                            #   (superseded by the reproducible codebase)
```

- **`scripts/`** — CLI entry points that wire a config file to the registries and run the attack or defense.
- **`image_datasets/`** — one subpackage per dataset, each with a config (`*_config.py`) and a download script (`*_download.py`) that fetches data via torchvision and writes images into the shared storage layout.
- **`attacks/`** — each attack subpackage defines how the (potentially poisoned) data loaders are built; `default_trainer.py` and `model_saver.py` handle training and persisting models.
- **`defenses/`** — each defense subpackage implements `train_defense()` / `test_defense()`; `defense_results_saver.py` exports `results.csv` and `artifact.json`.
- **`architectures/`** — model factory definitions selectable by name.
- **`registries/`** — name-to-implementation maps that let configs reference attacks, defenses, architectures, and datasets by string key, making it easy to add new ones.
- **`schema/`** — typed config objects parsed from the YAML files in `experiment_configs/`.
- **`experiment_configs/`** — YAML files describing attack and defense runs (dataset, architecture, counts, hyperparameters, etc.).
- **`utils/`** — cross-cutting helpers for deterministic seeding, the on-disk storage layout, and reproducibility bookkeeping.
- **`legacy_notebooks/`** — the original notebooks used to produce the paper's results. They are kept for reference but are no longer the primary workflow.