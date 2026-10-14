# SeDiR: Independent Reproduction

Unofficial reproduction of [*A Semantically Disentangled Unified Model for Multi-category 3D Anomaly Detection*](https://arxiv.org/abs/2603.25159). Built on the public MC3D-AD framework, it implements CFGT, C3L, GGD, and anomaly scoring. Unspecified details are documented in [the experimental report](docs/result.md).

## Results

Real3D-AD, one model across 12 categories; category-mean AUROC (%):

| Metric | SeDiR paper | This reproduction | Difference |
| --- | ---: | ---: | ---: |
| Object-level AUROC (O-AUROC) | 81.0 | 76.9 | −4.1 |
| Point-level AUROC (P-AUROC) | 80.6 | 78.9 | −1.7 |

See [results and protocol](docs/result.md) for per-category scores and implementation choices. Anomaly-ShapeNet results are not yet available.

## Requirements

Reported environment: Python 3.8.20, PyTorch 1.13.0, CUDA 11.7, Open3D 0.17.0.

```bash
pip install -r requirements.txt
```

Install `pointnet2_ops` separately for your PyTorch/CUDA versions. Datasets, Point-MAE weights, and SeDiR checkpoints are not included.

## Datasets

### Real3D-AD

Download and extract [Real3D-AD](https://drive.google.com/file/d/1oM4qjhlIMsQc_wiFIFIVBvuuR8nyk2k0/view?usp=sharing). From the repository root, downsample the training templates to 150,000 points with PointNet++ farthest-point sampling:

```bash
python tools/downsample_real3d.py --real3d_path /path/to/Real3D-AD-PCD
```

This creates `real3D_down` beside `Real3D-AD-PCD`. Set `dataset.data_dir` in the [Real3D-AD configuration](experiments/real3d/config_public.yaml) to the absolute path of `real3D_down`. Evaluation reads scans from the sibling `Real3D-AD-PCD` directory.

### Anomaly-ShapeNet

Download [Anomaly-ShapeNet](https://huggingface.co/datasets/Chopper233/Anomaly-ShapeNet) and extract `Anomaly-ShapeNet-v2/dataset/pcd`. Rename `pcd` to `Anomaly_ShapeNet` (underscore and case are required by this data loader):

```text
Anomaly_ShapeNet/
├── ashtray0/
│   ├── train/*.pcd
│   ├── test/*.pcd
│   └── GT/*.txt
├── bag0/
└── ...
```

Set `dataset.data_dir` in the [Anomaly-ShapeNet configuration](experiments/Anomaly_ShapeNet/config_public.yaml) to that directory's absolute path.

### Configuration

In either public configuration, `dataset.data_dir` also supplies the backbone's `data_dir` through a YAML alias; enter the dataset path once. Set the backbone's `checkpoint_path` to ModelNet40 8k pretrained Point-MAE weights. The manifests under `data/` use paths relative to the dataset root.

## Training and evaluation

From the repository root, train Real3D-AD on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_val.py --config experiments/real3d/config_public.yaml
```

For Anomaly-ShapeNet, use:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_val.py --config experiments/Anomaly_ShapeNet/config_public.yaml
```

Defaults: batch size 1, 1,000 epochs, AdamW at `1e-4`, and cosine annealing. Outputs are stored under the selected experiment directory.

For evaluation, set `saver.load_path` to a checkpoint from this implementation, then run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_val.py --config experiments/real3d/config_public.yaml --evaluate
```

Use the Anomaly-ShapeNet configuration path to evaluate that dataset. MC3D-AD reconstruction checkpoints are incompatible with this implementation.
