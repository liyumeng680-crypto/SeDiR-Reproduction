# SeDiR Reproduction: Experimental Results

This document reports the results and experimental settings of an independent, unofficial reproduction of *A Semantically Disentangled Unified Model for Multi-category 3D Anomaly Detection* (SeDiR). The implementation uses the public MC3D-AD project as its engineering foundation and implements the SeDiR-specific CFGT, C3L, GGD, and anomaly-scoring components from the paper. Details that are not fully specified in the paper are recorded below as implementation choices, not as claims about the authors' reference code.

## Real3D-AD evaluation protocol

The reported experiment trains one model across all 12 Real3D-AD categories. It uses four normal, full-view training templates per category (48 samples in total), each downsampled to 150,000 points. Evaluation uses 1,206 single-view scans: 604 normal and 602 anomalous. Each category contributes 100 test scans except Chicken, which contributes 106. Training uses normal samples only, with category labels available to the classification and contrastive components.

Object-level AUROC (O-AUROC) and point-level AUROC (P-AUROC) are computed separately for each category and then averaged equally across the 12 categories. All values in the tables below are percentages.

## Aggregate results

| Metric | SeDiR paper | This reproduction | Difference |
| --- | ---: | ---: | ---: |
| O-AUROC | 81.0 | 76.9 | −4.1 |
| P-AUROC | 80.6 | 78.9 | −1.7 |

As a check on the evaluation pipeline, the reproduced MC3D-AD baseline achieved 78.0 O-AUROC and 75.3 P-AUROC with the same pipeline; the corresponding reported values are 78.2 and 76.8. This comparison provides context for the evaluation procedure, but it does not establish equivalence with the unavailable SeDiR reference implementation.

## Per-category results

| Category | Paper O-AUROC | Reproduction O-AUROC | Paper P-AUROC | Reproduction P-AUROC |
| --- | ---: | ---: | ---: | ---: |
| Airplane | 86.0 | 87.0 | 66.6 | 57.8 |
| Car | 78.3 | 62.2 | 90.8 | 87.4 |
| Candybar | 81.9 | 72.0 | 95.1 | 93.7 |
| Chicken | 72.9 | 67.5 | 63.2 | 52.6 |
| Diamond | 94.8 | 95.4 | 97.7 | 97.1 |
| Duck | 86.2 | 86.3 | 83.4 | 86.6 |
| Fish | 93.8 | 90.6 | 96.7 | 95.9 |
| Gemstone | 62.7 | 57.6 | 54.7 | 54.0 |
| Seahorse | 67.4 | 67.2 | 73.9 | 75.2 |
| Shell | 77.9 | 76.4 | 79.8 | 84.0 |
| Starfish | 85.4 | 74.8 | 70.9 | 66.5 |
| Toffees | 84.5 | 86.1 | 94.0 | 95.5 |
| **Category mean** | **81.0** | **76.9** | **80.6** | **78.9** |

## Reproduction settings

### Data and local features

- PCD inputs are centered by subtracting the centroid. They are neither rescaled nor normalized to a unit sphere.
- Each sample is registered to a template from its category with Open3D FPFH and RANSAC (`voxel_size = 0.5`). Point order is preserved after registration.
- The local encoder is Point-MAE initialized from ModelNet40 8k pretraining. It remains frozen in evaluation mode, including its BatchNorm statistics.
- Features are taken from Transformer block index 3 (the fourth block), followed by LayerNorm, yielding 384 channels.
- Farthest-point sampling selects 1,024 shared centers. Neighborhoods contain 128, 256, or 512 points per center.

### CFGT and C3L

- CFGT uses a learned ACT token and a shared four-layer, eight-head context encoder with a 1,024-dimensional feed-forward layer, dropout of 0.1, post-layer normalization, and ReLU activation.
- Feature jitter is applied before the context encoder with `scale = 20.0` and `probability = 1.0`. Base, fine, and coarse features share the encoder. The global feature concatenates pooled base, coarse, and fine features with the encoded ACT token.
- The projected global token is L2-normalized and has 256 dimensions. The category classifier covers 12 classes.
- C3L uses a FIFO queue of 64 historical `(feature, category)` pairs. Supervised contrastive logits use temperature `0.07`; same-category entries are positives, and anchors without a positive entry are skipped. Current-batch features are enqueued after the optimizer step and are not used as keys for that step.
- The implemented objective is `L_rec + 0.001 L_scl + 0.001 L_cls + 0.01 L_cos`.

### Geometry-guided decoding

- Each decoder query combines the projected global token with the positional embedding of its group center. Encoded base features provide keys and values.
- Geometry is estimated from the registered input points. An adaptive neighborhood radius is set to seven times the mean nearest-neighbor distance. PCA-derived normals and curvature are summarized by local changes in normal angle and curvature.
- The two descriptor channels are min–max normalized across group centers and passed through a two-layer MLP with a hidden width of 32. The resulting additive attention bias uses a geometry weight of `1.0`.
- Reconstruction is trained against frozen base-scale Point-MAE features using the mean squared L2 error over group centers.

### Optimization and environment

| Setting | Value |
| --- | --- |
| Optimizer | AdamW; learning rate `1e-4`; betas `(0.9, 0.999)`; weight decay `1e-4` |
| Schedule | Cosine annealing for 1,000 epochs; minimum learning rate `0`; no warm-up |
| Batch size | 1 |
| Gradient clipping | Maximum norm `0.1` |
| Trainable parameters | 9,313,549 (Point-MAE frozen) |
| Hardware | One NVIDIA RTX 3090 with 24 GB of memory |
| Software | Python 3.8.20; PyTorch 1.13.0; CUDA 11.7; Open3D 0.17.0 |

## Anomaly scoring

The L2 distance between reconstructed and frozen base-scale features produces 1,024 token scores per sample. Each original point receives the score of its nearest sampled center. A one-dimensional Gaussian filter is then applied along the original point sequence, with a 511-point kernel, standard deviation `51`, and replicate padding. The object score is the maximum smoothed point score; the smoothed scores also support point-level AUROC. Min–max normalization is applied over the evaluated score set, not independently to each sample. This normalization is monotonic and therefore does not change AUROC.

The paper states a Gaussian parameter of `σ = 0.2` without defining its unit. This reproduction interprets it as `0.2 × (511 // 2) = 51`. The interpolation order, filtering domain, and normalization scope are explicit implementation choices.

## Interpretation

Several implementation details are not fully determined by the paper, including the contrastive temperature and queue update procedure, construction of decoder queries from a single global token, geometric neighborhood radius, descriptor normalization, and score post-processing. The paper's appendix and Table S7 also describe different assignments for the classification and cosine-loss weights; this reproduction uses `λ_scl = 0.001`, `λ_cls = 0.001`, and `λ_cos = 0.01`. These choices may contribute to the differences above, but the current results do not isolate their individual effects.

The repository also provides an Anomaly-ShapeNet configuration. No Anomaly-ShapeNet benchmark result is reported here.
