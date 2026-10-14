"""
SeDiR: CFGT + category-conditioned contrastive queue + geometry-guided decoder.
Replacement for models/reconstructions/uniad.py.

Paper equations (1)-(19) are implemented with explicit choices for unspecified
details. Read README_zh.md before treating this as an exact experimental replica.
No MC3D-AD random geometry mask is used.
"""
import copy
import hashlib
import warnings
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch import nn


def _get(cfg, key, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class GeometryDescriptor:
    """Supplement B: radius-PCA normal/curvature and their local variations.

    Default neighborhoods are taken from ALL input points, not only FPS centers.
    Radius rule and outward normal orientation are inherited implementation
    choices, because the paper does not specify those details.
    """
    def __init__(self, radius_multiplier=7., radius=None, source="points",
                 cache_size=32, chunk_size=256):
        if source not in ("points", "centers"):
            raise ValueError("geometry_source must be 'points' or 'centers'.")
        if radius_multiplier <= 0 or (radius is not None and radius <= 0):
            raise ValueError("Geometry radius must be positive.")
        self.multiplier, self.radius, self.source = radius_multiplier, radius, source
        self.cache_size, self.chunk_size = max(0, int(cache_size)), int(chunk_size)
        if self.chunk_size < 1:
            raise ValueError("geometry_chunk_size must be positive.")
        self.cache = OrderedDict()

    def _adaptive_radius(self, points, tree):
        if self.radius is not None:
            return float(self.radius)
        sample = np.linspace(0, len(points) - 1, min(1000, len(points)), dtype=np.int64)
        d, _ = tree.query(points[sample], k=2)
        distances = d[:, 1]
        distances = distances[np.isfinite(distances) & (distances > 0)]
        if not len(distances):
            raise ValueError("Cannot estimate a radius from coincident points.")
        return max(self.multiplier * float(distances.mean()), 1e-10)

    def _compute(self, points, center_idx):
        points = np.asarray(points, dtype=np.float64)
        if len(points) < 3:
            raise ValueError("Geometry estimation requires at least three points.")
        tree = cKDTree(points)
        radius = self._adaptive_radius(points, tree)
        n = len(points)
        normals = np.zeros((n, 3), dtype=np.float64)
        curvature = np.zeros(n, dtype=np.float64)
        valid = np.zeros(n, dtype=bool)
        # Chunked, vectorized covariance accumulation; no N x N distance matrix.
        for start in range(0, n, self.chunk_size):
            stop = min(start + self.chunk_size, n)
            neighbors = tree.query_ball_point(points[start:stop], radius)
            lengths = np.asarray([len(ids) for ids in neighbors], dtype=np.int64)
            # Radius neighborhoods include the query itself, hence lengths >= 1.
            flat = np.concatenate(neighbors).astype(np.int64)
            offsets = np.r_[0, np.cumsum(lengths[:-1])]
            p = points[flat]
            means = np.add.reduceat(p, offsets, axis=0) / lengths[:, None]
            second = np.einsum("ni,nj->nij", p, p)
            covariance = (np.add.reduceat(second, offsets, axis=0)
                          / lengths[:, None, None]
                          - np.einsum("ni,nj->nij", means, means))
            # Population vs unbiased covariance differs by a scalar only;
            # eigenvectors and lambda_min/sum(lambda) are unchanged.
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, 0.)
            ok = (lengths >= 3) & (eigenvalues[:, 1] > 1e-15)
            normals[start:stop] = eigenvectors[:, :, 0]
            curvature[start:stop] = eigenvalues[:, 0] / np.maximum(
                eigenvalues.sum(1), 1e-15)
            valid[start:stop] = ok
        # Same outward-orientation convention as the supplied MC3D-AD code.
        outward = np.sum(normals * (points - points.mean(0)), axis=1)
        normals[outward < 0] *= -1.
        desc = np.zeros((len(center_idx), 2), dtype=np.float64)
        for start in range(0, len(center_idx), self.chunk_size):
            ids = center_idx[start:start + self.chunk_size]
            balls = tree.query_ball_point(points[ids], radius)
            lengths = np.asarray([len(v) for v in balls], dtype=np.int64)
            row = np.repeat(np.arange(len(ids)), lengths)
            flat = np.concatenate(balls).astype(np.int64)
            keep = valid[flat] & valid[ids[row]]
            count = np.bincount(row[keep], minlength=len(ids)).clip(min=1)
            dot = np.sum(normals[ids[row]] * normals[flat], 1)
            angle = np.arccos(np.clip(dot, -1., 1.))
            change = np.abs(curvature[ids[row]] - curvature[flat])
            desc[start:start + len(ids), 0] = np.bincount(
                row[keep], weights=angle[keep], minlength=len(ids)) / count
            desc[start:start + len(ids), 1] = np.bincount(
                row[keep], weights=change[keep], minlength=len(ids)) / count
        return np.asarray(desc, dtype=np.float32)

    @torch.no_grad()
    def __call__(self, inputs, center):
        if self.source == "points":
            if "registered_pointcloud" not in inputs or "center_idx" not in inputs:
                raise KeyError("Full-point geometry requires registered_pointcloud and "
                               "center_idx from the replacement pointmae.py.")
            clouds = inputs["registered_pointcloud"].detach().float().cpu().numpy()
            indices = inputs["center_idx"].detach().long().cpu().numpy()
        else:
            # Explicit approximation option, not the full-point Supplement B formula.
            clouds = center.detach().float().cpu().numpy()
            indices = np.tile(np.arange(center.shape[1]), (center.shape[0], 1))
        descriptors = []
        for points, idx in zip(clouds, indices):
            points = np.ascontiguousarray(points, dtype=np.float32)
            idx = np.ascontiguousarray(idx, dtype=np.int64)
            if idx.min() < 0 or idx.max() >= len(points):
                raise IndexError("center_idx does not refer to the supplied point cloud.")
            # A filename-only cache is WRONG after resampling, registration or augmentation.
            h = hashlib.sha1(points.tobytes())
            h.update(idx.tobytes())
            key = h.digest()
            if key in self.cache:
                value = self.cache.pop(key)
                self.cache[key] = value
            else:
                value = self._compute(points, idx)
                if self.cache_size:
                    self.cache[key] = value
                    while len(self.cache) > self.cache_size:
                        self.cache.popitem(last=False)
            descriptors.append(value)
        return torch.as_tensor(np.stack(descriptors), device=center.device,
                               dtype=torch.float32)


class ContextEncoderLayer(nn.Module):
    def __init__(self, dim, heads, ff_dim, dropout, activation, pre_norm):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        act = nn.ReLU if activation == "relu" else nn.GELU
        self.ffn = nn.Sequential(nn.Linear(dim, ff_dim), act(),
                                 nn.Dropout(dropout), nn.Linear(ff_dim, dim))
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(self, x, pos):
        if self.pre_norm:
            normalized = self.norm1(x)
            qk = normalized + pos
            x = x + self.drop1(self.attn(qk, qk, normalized, need_weights=False)[0])
            return x + self.drop2(self.ffn(self.norm2(x)))
        qk = x + pos
        x = self.norm1(x + self.drop1(self.attn(qk, qk, x, need_weights=False)[0]))
        return self.norm2(x + self.drop2(self.ffn(x)))


class SharedContextEncoder(nn.Module):
    def __init__(self, dim, heads, layers, ff_dim, dropout, activation, pre_norm):
        super().__init__()
        self.layers = nn.ModuleList([
            ContextEncoderLayer(dim, heads, ff_dim, dropout, activation, pre_norm)
            for _ in range(layers)])
        self.norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()

    def forward(self, x, pos):
        for layer in self.layers:
            x = layer(x, pos)
        return self.norm(x)


class ContrastiveQueue(nn.Module):
    """FIFO historical buffer, persisted in state_dict; current batch is NOT a key."""
    def __init__(self, size, dim, temperature):
        super().__init__()
        if size < 1 or temperature <= 0:
            raise ValueError("queue_size and temperature must be positive.")
        self.temperature = float(temperature)
        self.register_buffer("features", torch.zeros(size, dim))
        self.register_buffer("labels", torch.full((size,), -1, dtype=torch.long))
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def loss(self, z, labels):
        n = int(self.count.item())
        if not n:
            return z.sum() * 0., z.new_zeros(())
        # Clone prevents version-counter problems if a caller enqueues before backward.
        bank = self.features[:n].detach().clone()
        bank_labels = self.labels[:n].detach().clone()
        logits = z.float() @ bank.float().t() / self.temperature
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        positives = labels[:, None].eq(bank_labels[None, :])
        positive_count = positives.sum(1)
        valid = positive_count > 0
        per_anchor = -(log_prob * positives).sum(1) / positive_count.clamp_min(1)
        loss = (per_anchor * valid).sum() / valid.sum().clamp_min(1)
        return loss, valid.sum().to(z.dtype)

    @torch.no_grad()
    def enqueue(self, z, labels):
        z = F.normalize(z.detach().to(self.features), dim=-1)
        labels = labels.detach().to(self.labels).reshape(-1)
        capacity = len(self.labels)
        if len(z) >= capacity:
            self.features.copy_(z[-capacity:])
            self.labels.copy_(labels[-capacity:])
            self.ptr.zero_()
            self.count.fill_(capacity)
            return
        for f, label in zip(z, labels):
            pos = int(self.ptr.item())
            self.features[pos].copy_(f)
            self.labels[pos].copy_(label)
            self.ptr.fill_((pos + 1) % capacity)
        self.count.fill_(min(capacity, int(self.count.item()) + len(z)))


class GeometryAttention(nn.Module):
    """Eq. (10). B_geo is broadcast over queries and heads, on the KEY axis."""
    def __init__(self, dim, heads, dropout):
        super().__init__()
        if dim % heads:
            raise ValueError("hidden_dim must be divisible by nhead.")
        self.heads, self.head_dim = heads, dim // heads
        self.q_proj, self.k_proj = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.v_proj, self.out_proj = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, query, key, value, key_bias, beta):
        b, qlen, dim = query.shape
        klen = key.shape[1]

        def split(x, length):
            return x.reshape(b, length, self.heads, self.head_dim).transpose(1, 2)

        q = split(self.q_proj(query), qlen)
        k = split(self.k_proj(key), klen)
        v = split(self.v_proj(value), klen)
        logits = (q * self.head_dim ** -.5) @ k.transpose(-1, -2)
        logits = logits + beta * key_bias[:, None, None, :].to(logits)
        weights = self.attn_drop(logits.float().softmax(-1).to(v.dtype))
        out = (weights @ v).transpose(1, 2).reshape(b, qlen, dim)
        return self.out_proj(out)


class GeometryDecoderLayer(nn.Module):
    """Fig. 4 wiring: semantic branch -> refinement branch -> concat -> FFN.

    The paper omits the exact single-z -> G-query expansion.
    Here each center has query z + positional_embedding(center).
    """
    def __init__(self, dim, heads, ff_dim, dropout, activation):
        super().__init__()
        self.semantic_attn = GeometryAttention(dim, heads, dropout)
        self.refine_attn = GeometryAttention(dim, heads, dropout)
        self.norm1, self.norm2, self.norm3 = [nn.LayerNorm(dim) for _ in range(3)]
        self.drop = nn.Dropout(dropout)
        act = nn.ReLU if activation == "relu" else nn.GELU
        self.ffn = nn.Sequential(
            nn.Linear(2 * dim, ff_dim), act(), nn.Dropout(dropout),
            nn.Linear(ff_dim, dim))

    def forward(self, previous, memory, semantic, pos, bias, beta):
        query = semantic.unsqueeze(1).expand(-1, memory.shape[1], -1) + pos
        left = self.norm1(query + self.drop(
            self.semantic_attn(query, memory + pos, memory, bias, beta)))
        right = self.norm2(left + self.drop(
            self.refine_attn(left + pos, previous + pos, previous, bias, beta)))
        return self.norm3(right + self.drop(self.ffn(torch.cat([left, right], -1))))


class GeometryGuidedDecoder(nn.Module):
    def __init__(self, dim, z_dim, heads, layers, ff_dim, dropout, activation):
        super().__init__()
        self.z_to_hidden = nn.Linear(z_dim, dim) if z_dim != dim else nn.Identity()
        self.layers = nn.ModuleList([
            GeometryDecoderLayer(dim, heads, ff_dim, dropout, activation)
            for _ in range(layers)])

    def forward(self, memory, z, pos, bias, beta):
        previous = memory  # Explicit initialization for Fig. 4's F_hat_(n-1).
        semantic = self.z_to_hidden(z)
        for layer in self.layers:
            previous = layer(previous, memory, semantic, pos, bias, beta)
        return previous


class GaussianTokenScorer(nn.Module):
    """Eq. (13): discrete 1-D Gaussian filter (kg=511, sigma=0.2) over the tokens.

    Norm (min-max) is applied over the whole score set at evaluation time
    (tools/train_val.py), NOT per instance: a per-instance min-max maps every
    sample's maximum to exactly 1, so S_obj = max(S_p) becomes the constant 1
    and object-level AUROC collapses to 0.5. This matches SeDiR Sec. 3.4 and
    Fig. 3, where Gaussian pooling is followed by normalization.
    The 'instance_minmax' switch is kept only to reproduce that failure mode.

    grid='index' is the literal discrete-offset interpretation of sigma=0.2.
    grid='normalized' is an OPTIONAL, non-verified interpretation of sigma=0.2.
    """
    def __init__(self, kernel_size=511, sigma=.2, grid="index",
                 normalization="none"):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 != 1 or sigma <= 0:
            raise ValueError("Gaussian kernel_size must be odd and sigma positive.")
        if grid not in ("index", "normalized"):
            raise ValueError("gaussian_grid must be 'index' or 'normalized'.")
        if normalization not in ("instance_minmax", "none"):
            raise ValueError("Unknown score_normalization.")
        half = kernel_size // 2
        x = torch.arange(-half, half + 1, dtype=torch.float32)
        if grid == "normalized" and half:
            x = x / half
        kernel = torch.exp(-.5 * (x / sigma).square())
        self.register_buffer("kernel", (kernel / kernel.sum()).reshape(1, 1, -1))
        self.padding, self.normalization = half, normalization

    def forward(self, raw):
        scores = raw.float()
        if self.normalization == "instance_minmax":
            low, high = scores.amin(-1, keepdim=True), scores.amax(-1, keepdim=True)
            scores = (scores - low) / (high - low).clamp_min(1e-12)
        scores = F.pad(scores, (self.padding, self.padding), mode="replicate")
        scores = F.conv1d(scores, self.kernel.to(scores))
        return scores, scores.flatten(1).amax(1)


class UniAD(nn.Module):
    """Retains the original class name so ModelHelper can build the new model."""
    def __init__(self, feature_size=1024, feature_jitter=None, neighbor_mask=None,
                 hidden_dim=256, initializer=None, cls_num=12, inplanes=384,
                 k=7, mask_ratio=.4, nhead=8, num_encoder_layers=4,
                 num_decoder_layers=4, dim_feedforward=1024, dropout=.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, z_dim=None, queue_size=64,
                 temperature=.07, lambda_scl=.001, lambda_cls=.001,
                 lambda_cos=.01, geometry_beta=1., geometry_hidden_dim=32,
                 geometry_source="points", geometry_radius=None,
                 geometry_cache_size=32, geometry_chunk_size=256,
                 gaussian_kernel_size=511, gaussian_sigma=.2,
                 gaussian_grid="index", score_normalization="none"):
        super().__init__()
        if activation not in ("relu", "gelu"):
            raise ValueError("activation must be relu or gelu.")
        if num_encoder_layers < 1 or num_decoder_layers < 1:
            raise ValueError("Encoder and decoder require at least one layer.")
        if hidden_dim % nhead:
            raise ValueError("hidden_dim must be divisible by nhead.")
        if neighbor_mask:
            warnings.warn("Legacy neighbor_mask/mask_ratio are ignored: SeDiR uses "
                          "a learned additive geometry bias, NOT random masking.")
        if return_intermediate_dec:
            warnings.warn("Only the final decoder layer is supervised/returned.")
        if score_normalization == "instance_minmax":
            warnings.warn("score_normalization='instance_minmax' reproduces a known "
                          "degenerate scoring: per-instance min-max maps every sample's "
                          "maximum to 1, so S_obj = max(S_p) is constant and object-level "
                          "AUROC collapses to 0.5. Eq. (13) normalizes over the score set.")
        self.feature_size, self.inplanes = int(feature_size), int(inplanes)
        self.cls_num = int(cls_num)
        self.jitter_enabled = feature_jitter is not False
        self.jitter_scale = float(_get(feature_jitter, "scale", 20.))
        self.jitter_prob = float(_get(feature_jitter, "prob", 1.))
        self.lambda_scl, self.lambda_cls, self.lambda_cos = lambda_scl, lambda_cls, lambda_cos
        self.geometry_beta = float(geometry_beta)
        z_dim = hidden_dim if z_dim is None else int(z_dim)
        self.input_proj = nn.Linear(inplanes, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, inplanes)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(inplace=True), nn.Linear(128, hidden_dim))
        self.act_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.encoder = SharedContextEncoder(
            hidden_dim, nhead, num_encoder_layers, dim_feedforward,
            dropout, activation, normalize_before)
        self.cls_head_finetune = nn.Sequential(
            nn.Linear(4 * hidden_dim, 256), nn.LayerNorm(256), nn.ReLU(inplace=True),
            nn.Dropout(.5), nn.Linear(256, 256), nn.LayerNorm(256),
            nn.ReLU(inplace=True), nn.Dropout(.5), nn.Linear(256, cls_num))
        self.projector = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, z_dim))
        self.queue = ContrastiveQueue(queue_size, z_dim, temperature)
        self.geometry = GeometryDescriptor(
            radius_multiplier=k, radius=geometry_radius, source=geometry_source,
            cache_size=geometry_cache_size, chunk_size=geometry_chunk_size)
        self.geo_mlp = nn.Sequential(
            nn.Linear(2, geometry_hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(geometry_hidden_dim, 1))
        self.decoder = GeometryGuidedDecoder(
            hidden_dim, z_dim, nhead, num_decoder_layers, dim_feedforward,
            dropout, activation)
        self.scorer = GaussianTokenScorer(
            gaussian_kernel_size, gaussian_sigma, gaussian_grid, score_normalization)
        self._initialize(initializer)
        nn.init.normal_(self.act_token, std=.02)

    def _initialize(self, cfg):
        method = _get(cfg, "method", "xavier_uniform")
        if method not in ("xavier_uniform", "xavier_normal", "normal",
                          "msra_uniform", "msra_normal"):
            raise ValueError("Unsupported initializer: " + str(method))

        def initialize_weight(weight):
            if method == "xavier_uniform":
                nn.init.xavier_uniform_(weight)
            elif method == "xavier_normal":
                nn.init.xavier_normal_(weight)
            elif method == "normal":
                nn.init.normal_(weight, std=float(_get(cfg, "std", .01)))
            elif method == "msra_uniform":
                nn.init.kaiming_uniform_(weight, a=1.)
            else:
                nn.init.kaiming_normal_(weight, a=1.)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                initialize_weight(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                initialize_weight(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)

    def add_jitter(self, tokens):
        if (self.training and self.jitter_enabled
                and torch.rand((), device=tokens.device).item() < self.jitter_prob):
            # Preserve the supplied MC3D-AD convention: ||feature|| / channel_count.
            scale = tokens.norm(dim=-1, keepdim=True) / tokens.shape[-1]
            tokens = tokens + torch.randn_like(tokens) * scale * self.jitter_scale
        return tokens

    def _tokens(self, feature):
        # The local extractor is fixed; the shared CFGT encoder is trainable.
        return self.input_proj(self.add_jitter(feature.detach().transpose(1, 2)))

    def _category_labels(self, inputs, batch_size, device):
        if "cls_label" not in inputs:
            if self.training:
                raise KeyError("SeDiR training needs cls_label (object category), "
                               "NOT label (normal/anomalous).")
            return None
        labels = torch.as_tensor(inputs["cls_label"], device=device)
        if labels.ndim == 2 and labels.shape[1] == self.cls_num and self.cls_num > 1:
            labels = labels.argmax(1)
        labels = labels.reshape(-1).long()
        if len(labels) != batch_size or (labels < 0).any() or (labels >= self.cls_num).any():
            raise ValueError("cls_label must contain B category IDs in [0, cls_num-1].")
        return labels

    def forward(self, inputs):
        required = ("xyz_features", "xyz_features_fine", "xyz_features_coarse", "center")
        missing = [key for key in required if key not in inputs]
        if missing:
            raise KeyError("Replace pointmae.py too. Missing multi-scale inputs: %s" % missing)
        base = inputs["xyz_features"].detach()
        fine, coarse = inputs["xyz_features_fine"], inputs["xyz_features_coarse"]
        b, channels, groups = base.shape
        if channels != self.inplanes or groups != self.feature_size:
            raise ValueError("Feature shape %s disagrees with inplanes=%d, feature_size=%d."
                             % (tuple(base.shape), self.inplanes, self.feature_size))
        if fine.shape != base.shape or coarse.shape != base.shape:
            raise ValueError("All three resolutions must use the SAME centers/channels.")
        center = inputs["center"].to(device=base.device, dtype=base.dtype)
        labels = self._category_labels(inputs, b, base.device)
        pos = self.pos_embed(center)
        act = self.act_token.expand(b, -1, -1)
        act_pos = torch.zeros_like(act)
        base_act = self.encoder(torch.cat([act, self._tokens(base)], 1),
                                torch.cat([act_pos, pos], 1))
        encoded_act, encoded_base = base_act[:, 0], base_act[:, 1:]
        encoded_fine = self.encoder(self._tokens(fine), pos)
        encoded_coarse = self.encoder(self._tokens(coarse), pos)
        # Eq. (4): SUM over the two non-base scales, mean over batch and groups.
        loss_cos = ((1. - F.cosine_similarity(encoded_base, encoded_fine, dim=-1))
                    + (1. - F.cosine_similarity(encoded_base, encoded_coarse, dim=-1))).mean()
        # Eq. (5), order is base, coarse, fine, ACT.
        global_feature = torch.cat([
            encoded_base.mean(1), encoded_coarse.mean(1),
            encoded_fine.mean(1), encoded_act], -1)
        cls_pred = self.cls_head_finetune(global_feature)  # NO detach here.
        z = F.normalize(self.projector(global_feature), dim=-1)
        loss_cls = F.cross_entropy(cls_pred, labels) if labels is not None else z.sum() * 0.
        if self.training:
            loss_scl, valid_anchors = self.queue.loss(z, labels)
        else:
            loss_scl, valid_anchors = z.sum() * 0., z.new_zeros(())
        desc = self.geometry(inputs, center)
        low, high = desc.amin(1, keepdim=True), desc.amax(1, keepdim=True)
        desc_norm = (desc - low) / (high - low).clamp_min(1e-12)
        bias = self.geo_mlp(desc_norm.to(base.dtype)).squeeze(-1)  # [B,G], Eq. (19).
        decoded = self.decoder(encoded_base, z, pos, bias, self.geometry_beta)
        feature_rec = self.output_proj(decoded).transpose(1, 2).contiguous()
        error = feature_rec.float() - base.float()
        # Eq. (11): average over B,G, SUM over feature channels, not F.mse_loss default.
        loss_rec = error.square().sum(1).mean()
        loss = (loss_rec + self.lambda_scl * loss_scl
                + self.lambda_cls * loss_cls + self.lambda_cos * loss_cos)
        raw_pred = torch.linalg.vector_norm(error, dim=1, keepdim=True)
        # Scores are not objectives; do not retain an unnecessary convolution graph.
        with torch.no_grad():
            pred, object_score = self.scorer(raw_pred.detach())
        output = dict(
            feature_rec=feature_rec, feature_align=base, pred=pred,
            pred_raw=raw_pred.detach(), object_score=object_score,
            raw_object_score=raw_pred.detach().flatten(1).amax(1),
            cls_pred=cls_pred, global_token=z, global_feature=global_feature,
            loss=loss, loss_rec=loss_rec, loss_cls=loss_cls,
            loss_cos=loss_cos, loss_scl=loss_scl,
            contrastive_valid_anchors=valid_anchors,
            contrastive_queue_size=self.queue.count.detach().clone(),
            geometry_bias=bias)
        if labels is not None:
            output["category_labels"] = labels
        return output

    @torch.no_grad()
    def update_memory(self, outputs):
        """Call ONCE AFTER backward/optimizer.step. Never call during validation."""
        if not self.training:
            raise RuntimeError("The contrastive queue must not be updated in eval mode.")
        z = outputs["global_token"].detach()
        labels = outputs["category_labels"].detach()
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            world = dist.get_world_size()
            size = torch.tensor([len(z)], dtype=torch.long, device=z.device)
            sizes = [torch.zeros_like(size) for _ in range(world)]
            dist.all_gather(sizes, size)
            maximum = max(int(v.item()) for v in sizes)
            zp = z.new_zeros((maximum, z.shape[1]))
            yp = labels.new_full((maximum,), -1)
            zp[:len(z)], yp[:len(labels)] = z, labels
            zs = [torch.empty_like(zp) for _ in range(world)]
            ys = [torch.empty_like(yp) for _ in range(world)]
            dist.all_gather(zs, zp)
            dist.all_gather(ys, yp)
            z = torch.cat([v[:int(n.item())] for v, n in zip(zs, sizes)])
            labels = torch.cat([v[:int(n.item())] for v, n in zip(ys, sizes)])
        self.queue.enqueue(z, labels)
