"""
SeDiR training/evaluation entry point; replaces tools/train_val.py.

Keeps the repository's ModelHelper, dataset builder, config shape and metric names.
Does NOT call the old criterion_helper or old dump/merge_together: their loss/
postprocessing would change SeDiR's objective and Eq. (13).
"""
import argparse
import copy
import logging
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

# Works both from the repository root and experiments/<dataset>/.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.data_builder import build_dataloader
from easydict import EasyDict
from models.model_helper import ModelHelper

METHOD_ID = "SeDiR-independent-paper-v1"
config = None
args = None
logger = logging.getLogger("global_logger")


def distributed():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if distributed() else 0


def world_size():
    return dist.get_world_size() if distributed() else 1


def barrier():
    if distributed():
        dist.barrier()


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def reconstruction_module(model):
    modules = [m for m in unwrap(model).modules()
               if callable(getattr(m, "update_memory", None))]
    if len(modules) != 1:
        raise RuntimeError("Expected exactly one SeDiR reconstruction module.")
    return modules[0]


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = bool(deterministic)


def resolve_path(value, base):
    if not value:
        return value
    value = os.path.expandvars(os.path.expanduser(str(value)))
    return value if os.path.isabs(value) else os.path.abspath(os.path.join(base, value))


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        cfg = EasyDict(yaml.safe_load(handle))
    exp = os.path.dirname(os.path.abspath(path))
    cfg.exp_path = exp
    # Public MC3D-AD configs express these relative to the experiment directory.
    if cfg.dataset.get("data_dir"):
        cfg.dataset.data_dir = resolve_path(cfg.dataset.data_dir, exp)
    for split in ("train", "test"):
        if cfg.dataset.get(split):
            for key in ("meta_file", "data_dir"):
                if cfg.dataset[split].get(key):
                    cfg.dataset[split][key] = resolve_path(cfg.dataset[split][key], exp)
    for subnet in cfg.net:
        for key in ("data_dir", "checkpoint_path"):
            if subnet.kwargs.get(key):
                subnet.kwargs[key] = resolve_path(subnet.kwargs[key], exp)
    cfg.save_path = resolve_path(cfg.saver.get("save_dir", "checkpoints_sedir"), exp)
    cfg.log_path = resolve_path(cfg.saver.get("log_dir", "log_sedir"), exp)
    cfg.evaluator.eval_dir = resolve_path(cfg.evaluator.get("save_dir", "result_sedir"), exp)
    for key in ("load_path", "resume_model"):
        if cfg.saver.get(key):
            cfg.saver[key] = resolve_path(cfg.saver[key], exp)
    return cfg


def setup_logging():
    logger.setLevel(logging.INFO if rank() == 0 else logging.WARNING)
    logger.handlers.clear()
    stream = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if rank() == 0:
        os.makedirs(config.log_path, exist_ok=True)
        log_file = os.path.join(config.log_path, time.strftime("sedir_%Y%m%d_%H%M%S.log"))
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False


def make_writer():
    if rank() != 0:
        return None
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            logger.warning("TensorBoard unavailable; text logging remains enabled.")
            return None
    return SummaryWriter(os.path.join(config.log_path, "events_sedir"))


def configure_category_mapping(train_loader, val_loader):
    datasets = [loader.dataset for loader in (train_loader, val_loader) if loader is not None]
    available = [d for d in datasets if hasattr(d, "label_dict")]
    if not available:
        logger.warning("Dataset has no label_dict. Its cls_label mapping must be stable.")
        return None
    names = set()
    for dataset in available:
        names.update(dataset.label_dict.keys())
    mapping = {name: i for i, name in enumerate(sorted(names))}
    for dataset in available:
        dataset.label_dict = dict(mapping)
    return mapping


def make_unpadded_validation_loader(loader):
    """Avoid duplicate test samples from DistributedSampler padding."""
    if loader is None or world_size() == 1:
        return loader
    subset = Subset(loader.dataset, range(rank(), len(loader.dataset), world_size()))
    return DataLoader(
        subset, batch_size=loader.batch_size or 1, shuffle=False,
        num_workers=loader.num_workers, pin_memory=loader.pin_memory,
        collate_fn=loader.collate_fn, drop_last=False)


def freeze_layers(model, names):
    base = unwrap(model)
    for name in names:
        if not hasattr(base, name):
            raise KeyError("Unknown frozen layer: " + name)
        module = getattr(base, name)
        module.eval()
        module.requires_grad_(False)


def read_checkpoint(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or ckpt.get("method") != METHOD_ID:
        raise RuntimeError(
            "This is not a checkpoint produced by this SeDiR implementation. "
            "Do not resume from MC3D-AD's old reconstruction checkpoint. "
            "Use the Point-MAE pretrained weights only through checkpoint_path.")
    return ckpt


def rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(model, optimizer, scheduler, epoch, best, mapping, is_best):
    if rank() != 0:
        return
    os.makedirs(config.save_path, exist_ok=True)
    value = dict(
        method=METHOD_ID, epoch=epoch, state_dict=unwrap(model).state_dict(),
        optimizer=optimizer.state_dict(), lr_scheduler=scheduler.state_dict(),
        best_metric=best, class_to_idx=mapping, rng_state=rng_state())
    path = os.path.join(config.save_path, "ckpt.pth.tar")
    torch.save(value, path + ".tmp")
    os.replace(path + ".tmp", path)
    if is_best:
        shutil.copyfile(path, os.path.join(config.save_path, "ckpt_best.pth.tar"))


def train_one_epoch(train_loader, model, optimizer, lr_scheduler, epoch,
                    start_iter, tb_logger, criterion=None, frozen_layers=()):
    # criterion intentionally unused: loss is the FOUR-term objective inside UniAD.
    model.train()
    freeze_layers(model, frozen_layers)
    head = reconstruction_module(model)
    totals = np.zeros(5, dtype=np.float64)
    seen = 0
    end = time.time()
    keys = ["loss", "loss_rec", "loss_scl", "loss_cls", "loss_cos"]
    for i, inputs in enumerate(train_loader):
        if "label" in inputs and torch.as_tensor(inputs["label"]).ne(0).any():
            raise ValueError("SeDiR training must contain only normal samples (label=0).")
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = outputs["loss"]
        finite = torch.isfinite(loss.detach()).to(dtype=torch.int32)
        if distributed():
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError("A nonfinite training loss was found on at least one rank.")
        loss.backward()
        max_norm = config.trainer.get("clip_max_norm")
        if max_norm is not None and float(max_norm) > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], float(max_norm))
        optimizer.step()
        # Only historical detached embeddings enter the queue, after this update.
        head.update_memory(outputs)
        values = torch.stack([outputs[key].detach().float() for key in keys])
        if distributed():
            dist.all_reduce(values)
            values /= world_size()
        values = values.cpu().numpy()
        totals += values
        seen += 1
        step = start_iter + i + 1
        if step % int(config.trainer.get("print_freq_step", 10)) == 0 and rank() == 0:
            logger.info(
                "Epoch %d/%d step %d | total %.6f rec %.6f scl %.6f cls %.6f "
                "cos %.6f | queue %d valid %d | lr %.3g time %.2fs",
                epoch + 1, config.trainer.max_epoch, step, *values.tolist(),
                int(head.queue.count.item()), int(outputs["contrastive_valid_anchors"].item()),
                optimizer.param_groups[0]["lr"], time.time() - end)
        if tb_logger is not None:
            for key, value in zip(keys, values):
                tb_logger.add_scalar("train/" + key, float(value), step)
            tb_logger.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
        end = time.time()
    if not seen:
        raise RuntimeError("Empty training loader.")
    return dict(zip(keys, (totals / seen).tolist()))


def _to_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _gaussian_kernel1d(kernel_size, sigma):
    """Normalised 1-D Gaussian kernel (Eq. 13: kg = 511)."""
    kernel_size = int(kernel_size)
    if kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError("score_smoothing.kernel_size must be a positive odd integer.")
    if not float(sigma) > 0:
        raise ValueError("score_smoothing.sigma must be positive.")
    half = kernel_size // 2
    x = torch.arange(-half, half + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / float(sigma)).square())
    return (kernel / kernel.sum()).reshape(1, 1, -1)


def smooth_point_scores(scores, cfg=None):
    """Eq. (13) Gaussian pooling along the ORIGINAL point sequence.

    Applied after the token -> point interpolation (point domain), which is the
    order used by the inherited MC3D-AD pipeline
    (merge_together: fill_missing_values -> avg_pool1d(511) -> max).
    Smoothing along the FPS token order instead is not spatial: consecutive FPS
    centres sit ~170-250x the nearest-neighbour distance apart.
    """
    cfg = cfg or {}
    kind = str(cfg.get("type", "none")).lower()
    if kind in ("none", "off", "identity"):
        return np.asarray(scores, dtype=np.float32)
    if kind != "gaussian":
        raise ValueError("Unsupported score_smoothing.type: " + kind)
    kernel = _gaussian_kernel1d(cfg.get("kernel_size", 511), cfg.get("sigma", 51.0))
    values = torch.as_tensor(np.asarray(scores, dtype=np.float32)).reshape(1, 1, -1)
    pad = kernel.shape[-1] // 2
    values = F.pad(values, (pad, pad), mode="replicate")
    return F.conv1d(values, kernel).reshape(-1).numpy()


def dump_sedir(folder, outputs, offset):
    """Dump per-sample token scores; smoothing and max happen in merge_sedir (Eq. 13)."""
    names = outputs["filename"]
    names = [names] if isinstance(names, str) else list(names)
    categories = outputs["clsname"]
    categories = [categories] if isinstance(categories, str) else list(categories)
    token_scores = _to_numpy(outputs["pred"])[:, 0]
    clouds = _to_numpy(outputs["pointcloud"])
    masks = _to_numpy(outputs["mask"])
    indices = _to_numpy(outputs["center_idx"])
    labels = _to_numpy(outputs["label"]).reshape(-1)
    class_labels = _to_numpy(outputs["category_labels"]).reshape(-1)
    class_preds = _to_numpy(outputs["cls_pred"].argmax(-1)).reshape(-1)
    object_scores = _to_numpy(outputs["object_score"]).reshape(-1)
    for i, name in enumerate(names):
        filename = "r%04d_s%09d.npz" % (rank(), offset + i)
        np.savez_compressed(
            os.path.join(folder, filename),
            filename=str(name), clsname=str(categories[i]), pred=token_scores[i],
            object_score_tokenmax=object_scores[i], point_cloud=clouds[i], mask=masks[i],
            center_idx=indices[i], label=labels[i], cls_label=class_labels[i],
            cls_pred=class_preds[i])
    return len(names)


def merge_sedir(folder):
    smoothing = config.evaluator.get("score_smoothing", {})
    fileinfos, labels, object_scores, masks, preds, class_labels, class_preds, points = (
        [], [], [], [], [], [], [], [])
    for path in sorted(Path(folder).glob("*.npz")):
        with np.load(path, allow_pickle=False) as record:
            xyz = np.asarray(record["point_cloud"])
            if xyz.shape[-1] != 3 and xyz.shape[0] == 3:
                xyz = xyz.T
            idx = np.asarray(record["center_idx"]).reshape(-1).astype(np.int64)
            scores = np.asarray(record["pred"]).reshape(-1)
            mask = np.asarray(record["mask"]).reshape(-1)
            if len(idx) != len(scores) or len(mask) != len(xyz):
                raise ValueError("Token indices/scores or point masks are misaligned in " + str(path))
            if np.any(idx < 0) or np.any(idx >= len(xyz)):
                raise IndexError("center_idx no longer indexes the original point cloud.")
            # Rigid registration preserves original point indices and distances.
            # Eq. (13) in the order used by the inherited MC3D-AD pipeline:
            #   (1) token -> point interpolation with the nearest centre,
            #   (2) 1-D smoothing along the POINT sequence,
            #   (3) S_obj = max(S_p).
            _, nearest = cKDTree(xyz[idx]).query(xyz, k=1)
            point_score = smooth_point_scores(scores[nearest], smoothing)
            fileinfos.append(dict(filename=str(record["filename"]), clsname=str(record["clsname"])))
            labels.append(int(np.asarray(record["label"]).item()))
            object_scores.append(float(point_score.max()))
            masks.append(mask)
            preds.append(point_score)
            class_labels.append(int(np.asarray(record["cls_label"]).item()))
            class_preds.append(int(np.asarray(record["cls_pred"]).item()))
            points.append(xyz)
    return fileinfos, labels, object_scores, masks, preds, class_labels, class_preds, points


def performances_sedir(fileinfos, labels, object_scores, masks, preds, cls_labels, cls_preds):
    def auc(target, score, name):
        target = np.asarray(target).reshape(-1)
        score = np.asarray(score, dtype=np.float64).reshape(-1)
        if not np.isfinite(score).all():
            raise FloatingPointError("Nonfinite evaluation score: " + name)
        if len(np.unique(target)) < 2:
            logger.warning("AUROC undefined for a single-label evaluation subset: %s", name)
            return float("nan")
        # Eq. (13): "Norm denotes min-max normalization" over the score set, the same
        # convention as the inherited MC3D-AD get_auc. It is monotone, hence
        # AUROC-invariant, and guarded so constant scores give 0.5 instead of NaN.
        low, high = float(score.min()), float(score.max())
        if high > low:
            score = (score - low) / (high - low)
        return float(roc_auc_score(target, score))

    result = {}
    categories = sorted({info["clsname"] for info in fileinfos})
    for category in categories:
        ix = [i for i, item in enumerate(fileinfos) if item["clsname"] == category]
        result[category + "_obj-AUROC_auc"] = auc(
            [labels[i] for i in ix], [object_scores[i] for i in ix], category + "/object")
        result[category + "_pixel-AUROC_auc"] = auc(
            np.concatenate([masks[i] for i in ix]) > 0,
            np.concatenate([preds[i] for i in ix]), category + "/point")
        result[category + "_cls-ACC_auc"] = float(np.mean(
            [cls_labels[i] == cls_preds[i] for i in ix]))
    for metric in ("obj-AUROC", "pixel-AUROC", "cls-ACC"):
        values = np.asarray([result[c + "_" + metric + "_auc"] for c in categories])
        finite = np.isfinite(values)
        result["mean_" + metric + "_auc"] = (
            float(values[finite].mean()) if finite.any() else float("nan"))
    return result


@torch.no_grad()
def validate(val_loader, model):
    if val_loader is None:
        raise RuntimeError("No validation/test loader is configured.")
    previous_mode = model.training
    model.eval()
    # Uneven validation shards must NOT execute DDP forward buffer broadcasts.
    evaluation_model = unwrap(model)
    device = next(evaluation_model.parameters()).device
    temp_dir = os.path.join(config.evaluator.eval_dir, "_sedir_eval_tmp")
    if rank() == 0:
        os.makedirs(temp_dir, exist_ok=True)
        for path in Path(temp_dir).glob("*.npz"):
            path.unlink()
    barrier()
    sample_offset = 0
    stats = torch.zeros(2, dtype=torch.float64, device=device)
    for i, inputs in enumerate(val_loader):
        outputs = evaluation_model(inputs)
        n = dump_sedir(temp_dir, outputs, sample_offset)
        sample_offset += n
        stats[0] += outputs["loss_rec"].detach().double() * n
        stats[1] += n
        if rank() == 0 and (i + 1) % int(config.trainer.get("print_freq_step", 10)) == 0:
            logger.info("Validation step %d/%d", i + 1, len(val_loader))
    if distributed():
        dist.all_reduce(stats)  # Synchronous: do not read before an async op completes.
    barrier()
    result = {}
    if rank() == 0:
        if stats[1].item() == 0:
            raise RuntimeError("Empty validation dataset.")
        values = merge_sedir(temp_dir)
        result = performances_sedir(*values[:7])
        logger.info("Validation reconstruction loss %.6f, samples %d",
                    (stats[0] / stats[1]).item(), int(stats[1].item()))
        score = np.asarray(values[2])
        logger.info("Object-score range: [%.10g, %.10g]", score.min(), score.max())
        if np.ptp(score) < 1e-5:
            logger.warning("Object scores nearly constant. Check evaluator."
                           "score_smoothing and the score-normalisation scope; with the "
                           "MC3D-AD-compatible order this should not happen.")
        for key, value in sorted(result.items()):
            logger.info("%s: %.6f", key, value)
        if args.evaluate and config.evaluator.get("vis_compound"):
            from utils.vis_helper import visualize_compound
            visualize_compound(
                values[0], values[4], values[3], values[1], values[7],
                config.evaluator.vis_compound)
        if not config.evaluator.get("keep_eval_files", False):
            shutil.rmtree(temp_dir)
    barrier()
    model.train(previous_mode)
    if previous_mode:
        freeze_layers(model, config.frozen_layers)
    return result


def main():
    global config, args
    parser = argparse.ArgumentParser(description="SeDiR paper-based implementation")
    parser.add_argument("--config", default="./config.yaml")
    parser.add_argument("-e", "--evaluate", action="store_true")
    parser.add_argument("--local_rank", "--local-rank", dest="local_rank", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank or 0))
    requested_world = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if requested_world > 1:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo",
                                init_method="env://")
    setup_logging()
    seed = config.get("random_seed", 42)
    set_seed(42 if seed is None else int(seed), config.get("reproduce", False))
    logger.info("Configuration: %s", config)
    if config.get("criterion"):
        logger.warning("Legacy config.criterion is ignored. Using the four SeDiR losses.")

    model = ModelHelper(config.net)
    model.cuda() if torch.cuda.is_available() else model.cpu()
    # Union of both repository freezing conventions; never XOR.
    frozen = set(config.get("frozen_layers", []))
    frozen.update(item.name for item in config.net if item.get("frozen", False))
    config.frozen_layers = sorted(frozen)
    freeze_layers(model, config.frozen_layers)
    train_loader, val_loader = build_dataloader(
        copy.deepcopy(config.dataset), distributed=requested_world > 1)
    mapping = configure_category_mapping(train_loader, val_loader)
    head = reconstruction_module(model)
    if mapping is not None and len(mapping) != head.cls_num:
        raise ValueError("Dataset has %d categories, but cls_num=%d."
                         % (len(mapping), head.cls_num))
    val_loader = make_unpadded_validation_loader(val_loader)
    if requested_world > 1:
        use_cuda = torch.cuda.is_available()
        model = DDP(model, device_ids=[local_rank] if use_cuda else None,
                    output_device=local_rank if use_cuda else None,
                    find_unused_parameters=True, broadcast_buffers=False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise RuntimeError("There are no trainable parameters.")
    opt_cfg = config.trainer.get("optimizer", {})
    if _optimizer_type(opt_cfg) != "AdamW":
        logger.warning("Using AdamW as specified by the paper, instead of the old optimizer.")
    opt_kwargs = dict(opt_cfg.get("kwargs", {}))
    allowed = ("lr", "betas", "eps", "weight_decay", "amsgrad")
    opt_kwargs = {k: v for k, v in opt_kwargs.items() if k in allowed}
    opt_kwargs.setdefault("lr", 1e-4)
    opt_kwargs.setdefault("weight_decay", 1e-4)  # Inherited, not specified in the paper.
    optimizer = torch.optim.AdamW(parameters, **opt_kwargs)
    sched_cfg = config.trainer.get("lr_scheduler", {})
    if sched_cfg.get("type", "CosineAnnealingLR") != "CosineAnnealingLR":
        logger.warning("Replacing the old StepLR with the paper's cosine schedule.")
    eta_min = float(sched_cfg.get("kwargs", {}).get("eta_min", 0.))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config.trainer.max_epoch), eta_min=eta_min)
    best, start_epoch = -float("inf"), 0
    resume = config.saver.get("resume_model")
    latest = os.path.join(config.save_path, "ckpt.pth.tar")
    if config.saver.get("auto_resume", False) and os.path.isfile(latest):
        resume = latest
    load_path = resume or config.saver.get("load_path")
    if args.evaluate and not load_path:
        raise ValueError("Evaluation requires saver.load_path or a SeDiR resume_model.")
    if load_path:
        checkpoint = read_checkpoint(load_path)
        old_mapping = checkpoint.get("class_to_idx")
        if old_mapping is not None and mapping is not None and old_mapping != mapping:
            raise ValueError("Checkpoint and dataset category-to-ID mappings differ.")
        state = {k.removeprefix("module.") if hasattr(k, "removeprefix")
                 else (k[7:] if k.startswith("module.") else k): v
                 for k, v in checkpoint["state_dict"].items()}
        unwrap(model).load_state_dict(state, strict=True)
        if resume and not args.evaluate:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["lr_scheduler"])
            start_epoch, best = checkpoint["epoch"], checkpoint["best_metric"]
            # Exact RNG restore is provided for the paper's single-process setup.
            if world_size() == 1:
                restore_rng(checkpoint.get("rng_state"))
    logger.info("Trainable parameters: %d", sum(p.numel() for p in parameters))
    if args.evaluate:
        validate(val_loader, model)
        if distributed():
            dist.destroy_process_group()
        return
    if train_loader is None:
        raise RuntimeError("No training loader.")
    writer = make_writer()
    val_freq = int(config.trainer.get("val_freq_epoch", 100))
    save_freq = int(config.trainer.get("save_freq_epoch", val_freq))
    if val_freq < 1 or save_freq < 1:
        raise ValueError("val_freq_epoch and save_freq_epoch must be positive.")
    key_metric = config.evaluator.get("key_metric", "mean_obj-AUROC_auc")
    for epoch in range(start_epoch, int(config.trainer.max_epoch)):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        train_one_epoch(
            train_loader, model, optimizer, scheduler, epoch,
            epoch * len(train_loader), writer, frozen_layers=config.frozen_layers)
        scheduler.step()
        do_validate = (epoch + 1) % val_freq == 0 or epoch + 1 == config.trainer.max_epoch
        is_best = False
        if do_validate:
            metrics = validate(val_loader, model)
            if rank() == 0:
                metric = metrics[key_metric]
                is_best = math.isfinite(metric) and metric >= best
                if is_best:
                    best = metric
                if writer is not None:
                    for key, value in metrics.items():
                        writer.add_scalar("validation/" + key, value, epoch + 1)
        if do_validate or (epoch + 1) % save_freq == 0:
            save_checkpoint(model, optimizer, scheduler, epoch + 1, best, mapping, is_best)
        barrier()
    if writer is not None:
        writer.close()
    if distributed():
        dist.destroy_process_group()


def _optimizer_type(cfg):
    return cfg.get("type", "AdamW")


if __name__ == "__main__":
    main()
