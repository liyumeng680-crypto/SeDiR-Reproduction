"""
SeDiR paper-based implementation: frozen, shared, multi-resolution Point-MAE.
Replacement for models/backbones/pointmae/pointmae.py.

The Point-MAE parameter names follow the user's MC3D-AD implementation.
feature_layers=(3,) preserves that implementation's effective feature output.
This is an independent implementation, NOT an official SeDiR code release.
"""
import hashlib
import os
import warnings
from collections import OrderedDict
from contextlib import nullcontext

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

try:
    from pointnet2_ops import pointnet2_utils
except ImportError:
    pointnet2_utils = None


def square_distance(src, dst):
    return (src.square().sum(-1, keepdim=True)
            + dst.square().sum(-1).unsqueeze(1)
            - 2 * src @ dst.transpose(-1, -2)).clamp_min(0)


def index_points(points, idx):
    shape = [points.shape[0]] + [1] * (idx.ndim - 1)
    batch = torch.arange(points.shape[0], device=points.device).view(*shape)
    return points[batch, idx.long()]


@torch.no_grad()
def fps(data, number):
    """Returns centers [B,G,3] and ORIGINAL point indices [B,G]."""
    data = data.contiguous().float()
    b, n, c = data.shape
    if c != 3 or not 0 < number <= n:
        raise ValueError("FPS requires [B,N,3] and 0 < num_group <= N.")
    if data.is_cuda and pointnet2_utils is not None:
        idx = pointnet2_utils.furthest_point_sample(data, number).long()
    else:
        # Deterministic CPU/testing fallback. CUDA extension is preferred at scale.
        idx = torch.empty(b, number, dtype=torch.long, device=data.device)
        distance = torch.full((b, n), float("inf"), device=data.device)
        farthest = torch.zeros(b, dtype=torch.long, device=data.device)
        batch = torch.arange(b, device=data.device)
        for i in range(number):
            idx[:, i] = farthest
            center = data[batch, farthest].unsqueeze(1)
            distance = torch.minimum(distance, (data - center).square().sum(-1))
            farthest = distance.argmax(-1)
    return index_points(data, idx), idx


class KNN(nn.Module):
    def __init__(self, k, transpose_mode=True):
        super().__init__()
        self.k, self.transpose_mode = int(k), transpose_mode

    @torch.no_grad()
    def forward(self, ref, query):
        if not self.transpose_mode:
            ref, query = ref.transpose(1, 2), query.transpose(1, 2)
        if ref.shape[1] < self.k:
            raise ValueError("The point cloud has fewer points than requested neighbors.")
        distances, indices = [], []
        for r, q in zip(ref.detach().cpu().numpy(), query.detach().cpu().numpy()):
            d, i = cKDTree(r).query(q, k=self.k)
            distances.append(np.asarray(d).reshape(len(q), self.k))
            indices.append(np.asarray(i).reshape(len(q), self.k))
        return (torch.as_tensor(np.stack(distances), device=ref.device,
                                dtype=ref.dtype),
                torch.as_tensor(np.stack(indices), device=ref.device,
                                dtype=torch.long))


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group, self.group_size = int(num_group), int(group_size)

    @torch.no_grad()
    def indices_multiscale(self, xyz):
        if self.group_size % 2 or self.group_size < 2:
            raise ValueError("The base group_size must be an even integer >= 2.")
        if xyz.shape[1] < max(self.num_group, 2 * self.group_size):
            raise ValueError(
                "Need N >= max(num_group, 2*group_size), got N=%d." % xyz.shape[1])
        center, center_idx = fps(xyz, self.num_group)
        # ONE FPS and ONE sorted 2k-NN query: all resolutions share centers.
        _, idx = KNN(2 * self.group_size)(xyz, center)
        return center, idx, center_idx

    def forward(self, xyz):
        center, center_idx = fps(xyz, self.num_group)
        _, idx = KNN(self.group_size)(xyz, center)
        return index_points(xyz, idx) - center.unsqueeze(2), center, idx, center_idx


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1))
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, encoder_channel, 1))

    def forward(self, groups):
        b, g, n, _ = groups.shape
        f = self.first_conv(groups.reshape(b * g, n, 3).transpose(1, 2))
        pooled = f.amax(2, keepdim=True).expand(-1, -1, n)
        return self.second_conv(torch.cat([pooled, f], 1)).amax(2).reshape(
            b, g, self.encoder_channel)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep = 1. - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (keep + torch.rand(shape, dtype=x.dtype, device=x.device)).floor()
        return x * mask / keep


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features or in_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features or in_features, out_features or in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads,
                                  c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = self.attn_drop(((q * self.scale) @ k.transpose(-1, -2)).softmax(-1))
        return self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(b, n, c)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.attn = Attention(dim, num_heads)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=384, depth=12, num_heads=6,
                 drop_path_rate=0.1, fetch_idx=(3,)):
        super().__init__()
        rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, drop_path=r) for r in rates])
        self.fetch_idx = tuple(fetch_idx)
        if not self.fetch_idx or sorted(set(self.fetch_idx)) != list(self.fetch_idx):
            raise ValueError("feature_layers must be sorted, unique, zero-based indices.")
        if not 0 <= self.fetch_idx[0] <= self.fetch_idx[-1] < depth:
            raise ValueError("Invalid feature_layers.")

    def forward(self, x, pos):
        features = []
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in self.fetch_idx:
                features.append(x)
            if i == self.fetch_idx[-1]:
                break  # Later unused frozen layers cannot affect the selected outputs.
        return features


def _trusted_torch_load(path):
    # Load only checkpoints from a source you trust.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch 1.13
        return torch.load(path, map_location="cpu")


class PointTransformer(nn.Module):
    def __init__(self, group_size=256, num_group=1024, encoder_dims=384,
                 feature_layers=(3,), encoder_group_chunk=64):
        super().__init__()
        self.trans_dim, self.depth, self.num_heads = 384, 12, 6
        self.group_size, self.num_group = int(group_size), int(num_group)
        self.encoder_dims = encoder_dims
        self.encoder_group_chunk = int(encoder_group_chunk)
        self.feature_layers = tuple(feature_layers)
        self.group_divider = Group(num_group, group_size)
        self.encoder = Encoder(encoder_dims)
        if encoder_dims != self.trans_dim:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            self.reduce_dim = nn.Linear(encoder_dims, self.trans_dim)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim))
        self.blocks = TransformerEncoder(fetch_idx=self.feature_layers)
        self.norm = nn.LayerNorm(self.trans_dim)
        # Retained solely for compatibility with the supplied Point-MAE checkpoint.
        self.cls_head_finetune = nn.Sequential(
            nn.Linear(768, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(.5), nn.Linear(256, 256), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.Dropout(.5), nn.Linear(256, 40))

    @property
    def out_channels(self):
        return self.trans_dim * len(self.feature_layers)

    def load_model_from_ckpt(self, path):
        if not path or not os.path.isfile(path):
            raise FileNotFoundError("Point-MAE checkpoint not found: %r" % path)
        ckpt = _trusted_torch_load(path)
        if not isinstance(ckpt, dict):
            raise TypeError("Expected a state_dict or a checkpoint dictionary.")
        state = ckpt
        for key in ("base_model", "model_state_dict", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
        own, mapped = self.state_dict(), {}
        prefixes = ("module.", "MAE_encoder.", "base_model.", "transformer_q.",
                    "backbone.xyz_backbone.", "xyz_backbone.")
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if key.startswith(prefix):
                        key = key[len(prefix):]
                        changed = True
                        break
            if key in own and own[key].shape == value.shape:
                mapped[key] = value
        required = []
        for key in own:
            used = key.startswith(("encoder.", "pos_embed.", "norm.", "reduce_dim."))
            if key.startswith("blocks.blocks."):
                used = int(key.split(".")[2]) <= max(self.feature_layers)
            used |= key in ("cls_token", "cls_pos")
            if used and not key.endswith("num_batches_tracked"):
                required.append(key)
        missing = [key for key in required if key not in mapped]
        if missing:
            raise RuntimeError(
                "Required Point-MAE weights are missing or have wrong shapes; "
                "refusing to freeze random weights. First missing keys: %s" % missing[:12])
        self.load_state_dict(mapped, strict=False)
        print("[Point-MAE] Loaded %d compatible tensors from %s" % (len(mapped), path))

    def load_model_from_pb_ckpt(self, path):
        self.load_model_from_ckpt(path)

    def _encode_resolution(self, xyz, center, idx):
        # Chunk the local Conv1d encoder to avoid G*512*512-sized activations.
        # Chunking is equivalent ONLY with frozen/eval BatchNorm.
        chunk = self.encoder_group_chunk if not self.training else self.num_group
        chunk = self.num_group if chunk <= 0 else chunk
        tokens = []
        for start in range(0, self.num_group, chunk):
            groups = index_points(xyz, idx[:, start:start + chunk])
            groups = groups - center[:, start:start + chunk].unsqueeze(2)
            tokens.append(self.encoder(groups))
        x = torch.cat(tokens, 1)
        pos = self.pos_embed(center)
        has_cls = self.encoder_dims != self.trans_dim
        if has_cls:
            x = self.reduce_dim(x)
            x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], 1)
            pos = torch.cat([self.cls_pos.expand(x.shape[0], -1, -1), pos], 1)
        values = self.blocks(x, pos)
        values = [self.norm(v)[:, int(has_cls):].transpose(1, 2) for v in values]
        return torch.cat(values, 1).contiguous()

    def forward_multiscale(self, pts):
        if pts.ndim != 3 or pts.shape[1] != 3:
            raise ValueError("PointTransformer expects [B,3,N].")
        xyz = pts.transpose(1, 2).contiguous().float()
        center, max_idx, center_idx = self.group_divider.indices_multiscale(xyz)
        k = self.group_size
        fine = self._encode_resolution(xyz, center, max_idx[:, :, :k // 2])
        base = self._encode_resolution(xyz, center, max_idx[:, :, :k])
        coarse = self._encode_resolution(xyz, center, max_idx)
        return dict(xyz_features=base, xyz_features_fine=fine,
                    xyz_features_coarse=coarse, center=center,
                    ori_idx=max_idx[:, :, :k].contiguous(), center_idx=center_idx)

    def forward(self, pts):
        out = self.forward_multiscale(pts)
        return out["xyz_features"], out["center"], out["ori_idx"], out["center_idx"]


def _open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("Registration requires open3d; install it or explicitly set "
                          "use_registration=False for pre-aligned inputs.") from exc
    return o3d


def get_first_file_in_subfolders(directory):
    root = os.path.abspath(os.path.expanduser(directory))
    templates = {}
    for category in sorted(os.listdir(root)):
        folder = os.path.join(root, category, "train")
        if not os.path.isdir(folder):
            continue
        paths = [os.path.join(folder, name) for name in sorted(os.listdir(folder))
                 if name.lower().endswith((".pcd", ".ply", ".xyz", ".pts"))]
        if paths:
            templates[category] = paths[0]  # Store absolute paths; do not join twice.
    return templates


def preprocess_point_cloud(pcd, voxel_size):
    o3d = _open3d()
    down = pcd.voxel_down_sample(voxel_size)
    if len(down.points) < 3:
        raise ValueError("Registration downsampling leaves <3 points; check voxel_size "
                         "and point coordinate units.")
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return down, fpfh


def get_registration_np(source_data, target_data, voxel_size=0.5, seed=0):
    """Preserves MC3D-AD rigid registration. Never changes the point order."""
    o3d = _open3d()
    if seed is not None and hasattr(o3d.utility, "random"):
        o3d.utility.random.seed(int(seed))
    source, target = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_data)
    target.points = o3d.utility.Vector3dVector(target_data)
    init = np.array([[0., 0., 1., 0.], [1., 0., 0., 0.],
                     [0., 1., 0., 0.], [0., 0., 0., 1.]])
    source.transform(init)
    sd, sf = preprocess_point_cloud(source, voxel_size)
    td, tf = preprocess_point_cloud(target, voxel_size)
    reg = o3d.pipelines.registration
    threshold = voxel_size * 1.5
    result = reg.registration_ransac_based_on_feature_matching(
        sd, td, sf, tf, True, threshold,
        reg.TransformationEstimationPointToPoint(False), 3,
        [reg.CorrespondenceCheckerBasedOnEdgeLength(.9),
         reg.CorrespondenceCheckerBasedOnDistance(threshold)],
        reg.RANSACConvergenceCriteria(100000, .999))
    source.transform(result.transformation)
    return np.asarray(source.points).astype(np.float32)


class Model1(nn.Module):
    def __init__(self, data_dir="", device=None, out_indices=None,
                 checkpoint_path="", xyz_backbone_name="Point_MAE",
                 group_size=256, num_group=1024, feature_layers=(3,),
                 encoder_group_chunk=64, use_registration=True,
                 registration_voxel_size=0.5, registration_seed=0,
                 registration_cache_size=16, allow_random_init=False,
                 pretrained=True, outlayers=None):
        super().__init__()
        if xyz_backbone_name != "Point_MAE":
            raise ValueError("This implementation expects a Point_MAE backbone.")
        self.xyz_backbone = PointTransformer(
            group_size, num_group, feature_layers=feature_layers,
            encoder_group_chunk=encoder_group_chunk)
        if checkpoint_path:
            self.xyz_backbone.load_model_from_ckpt(checkpoint_path)
        elif not allow_random_init:
            raise ValueError("A pretrained checkpoint is required. "
                             "allow_random_init=True is for smoke tests ONLY.")
        else:
            warnings.warn("Random Point-MAE weights: smoke test only, not a trained model.")
        self.xyz_backbone.requires_grad_(False)
        self.xyz_backbone.eval()
        self.use_registration = bool(use_registration)
        self.registration_voxel_size = registration_voxel_size
        self.registration_seed = registration_seed
        self.registration_cache_size = max(0, int(registration_cache_size))
        self._registration_cache, self._template_points = OrderedDict(), {}
        self.template = get_first_file_in_subfolders(data_dir) if use_registration else {}
        self.outstrides = [1]

    def get_outplanes(self):
        return self.xyz_backbone.out_channels

    def get_outstrides(self):
        return self.outstrides

    def train(self, mode=True):
        super().train(mode)
        self.xyz_backbone.eval()  # Frozen BatchNorm and DropPath, even after model.train().
        return self

    def norm_pcd(self, points):
        return points - points.mean(0, keepdims=True)

    def _register(self, xyz, categories):
        result = []
        for points, category in zip(xyz.detach().cpu().numpy(), categories):
            if category not in self.template:
                raise KeyError("No training template for category %r." % category)
            digest = hashlib.sha1(np.ascontiguousarray(points).tobytes()).hexdigest()
            key = (category, digest)
            if key in self._registration_cache:
                registered = self._registration_cache.pop(key)
                self._registration_cache[key] = registered
            else:
                if category not in self._template_points:
                    pcd = _open3d().io.read_point_cloud(self.template[category])
                    template = np.asarray(pcd.points).copy()
                    if not len(template):
                        raise ValueError("Empty registration template: " + self.template[category])
                    self._template_points[category] = self.norm_pcd(template)
                registered = get_registration_np(
                    points, self._template_points[category],
                    self.registration_voxel_size, self.registration_seed)
                if self.registration_cache_size:
                    self._registration_cache[key] = registered
                    while len(self._registration_cache) > self.registration_cache_size:
                        self._registration_cache.popitem(last=False)
            result.append(registered)
        return torch.as_tensor(np.stack(result), dtype=torch.float32, device=xyz.device)

    @torch.no_grad()
    def forward(self, pointcloud):
        device = next(self.xyz_backbone.parameters()).device
        xyz = torch.as_tensor(pointcloud["pointcloud"], device=device, dtype=torch.float32)
        if xyz.ndim != 3:
            raise ValueError("pointcloud must have shape [B,N,3] or [B,3,N].")
        if xyz.shape[-1] != 3:
            if xyz.shape[1] == 3:
                xyz = xyz.transpose(1, 2)
            else:
                raise ValueError("Invalid pointcloud shape.")
        if not torch.isfinite(xyz).all():
            raise ValueError("Point cloud contains NaN or Inf.")
        if self.use_registration:
            names = pointcloud["clsname"]
            names = [names] if isinstance(names, str) else list(names)
            if len(names) != xyz.shape[0]:
                raise ValueError("clsname batch length mismatch.")
            xyz = self._register(xyz, names)
        out = self.xyz_backbone.forward_multiscale(xyz.transpose(1, 2).contiguous())
        out["registered_pointcloud"] = xyz
        return out
