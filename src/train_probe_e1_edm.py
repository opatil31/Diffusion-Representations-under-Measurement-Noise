import os
import sys
import copy
import json
import math
import argparse
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Subset

import e1_data as E1
from train_probe_e1 import linear_probe, NUM_CLASSES, IMG_SIZE   # backbone-agnostic probe


def edm_spec(dataset):
    if dataset == "cifar10":
        return dict(model_channels=128, channel_mult=[2, 2, 2], num_blocks=4,
                    attn_resolutions=[16], dropout=0.13, resample_filter=[1, 1],
                    channel_mult_noise=1, embedding_type="positional",
                    lr=1e-3, batch=512, duration_mimg=200.0, augment=0.12)
    if dataset == "tiny_imagenet":            # 64x64; no official EDM config -> FFHQ/AFHQ-like
        return dict(model_channels=128, channel_mult=[1, 2, 2, 2], num_blocks=4,
                    attn_resolutions=[16], dropout=0.10, resample_filter=[1, 1],
                    channel_mult_noise=1, embedding_type="positional",
                    lr=2e-4, batch=256, duration_mimg=100.0, augment=0.0)
    raise ValueError(dataset)


@dataclass
class EDMConfig:
    dataset: str = "cifar10"
    edm_repo: str = "./edm"
    out_dir: str = "./e1_runs_edm"
    params_json: str = "./e1_artifacts/e1_noise_params.json"
    conditions: List[tuple] = field(default_factory=lambda: [
        ("clean", 0),
        ("gaussian", 1), ("poisson_gaussian", 1), ("speckle", 1),
    ])
    duration_mimg: Optional[float] = None
    batch_size: Optional[int] = None
    max_steps: Optional[int] = None
    lr_rampup_kimg: float = 10000.0
    ema_halflife_kimg: float = 500.0
    ema_rampup_ratio: float = 0.05
    use_augment: Optional[bool] = None        # None => follow edm_spec's augment>0
    fp16: bool = False                         # AMP for speed on large GPUs

    probe_sigmas: tuple = (0.05, 0.1, 0.2, 0.5)
    probe_layers: Optional[tuple] = None
    probe_epochs: int = 60
    probe_lr: float = 1e-3
    probe_batch: int = 256
    probe_feat_subset: int = 0

    num_workers: int = 4
    log_every: int = 1000
    ckpt_every: int = 20000                    # full train-state checkpoint interval (steps)
    resume: bool = True                        # auto-resume an interrupted condition from its train-state
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_net(cfg, spec, img_ch, img_size, augment_on, use_fp16=False):
    from training.networks import EDMPrecond
    return EDMPrecond(
        img_resolution=img_size, img_channels=img_ch, label_dim=0,
        use_fp16=use_fp16,                       # EDM-native mixed precision (no torch.autocast)
        model_type="SongUNet",
        model_channels=spec["model_channels"], channel_mult=spec["channel_mult"],
        num_blocks=spec["num_blocks"], attn_resolutions=spec["attn_resolutions"],
        dropout=spec["dropout"], augment_dim=9 if augment_on else 0,
        embedding_type=spec["embedding_type"], encoder_type="standard",
        decoder_type="standard", channel_mult_noise=spec["channel_mult_noise"],
        resample_filter=spec["resample_filter"],
    )


class FeatureTaps:
    def __init__(self, net):
        from training.networks import UNetBlock
        self.feats, self.handles, self.names = [], [], []
        for name, m in net.model.dec.items():
            if isinstance(m, UNetBlock):
                self.names.append(name)
                self.handles.append(m.register_forward_hook(self._hook()))

    def _hook(self):
        def fn(_m, _i, out):
            self.feats.append(out)
        return fn

    def clear(self):
        self.feats = []

    def remove(self):
        for h in self.handles:
            h.remove()


def train_edm(cfg, spec, train_dataset, img_ch, img_size, tag):
    from training.loss import EDMLoss
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    dev = cfg.device
    augment_on = spec["augment"] > 0 if cfg.use_augment is None else cfg.use_augment

    net = build_net(cfg, spec, img_ch, img_size, augment_on, use_fp16=cfg.fp16).to(dev).train()
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(net.parameters(), lr=spec["lr"], betas=(0.9, 0.999), eps=1e-8)
    loss_fn = EDMLoss(P_mean=-1.2, P_std=1.2, sigma_data=0.5)

    augment_pipe = None
    if augment_on:
        from training.augment import AugmentPipe
        augment_pipe = AugmentPipe(p=spec["augment"], xflip=1e8, yflip=1, scale=1,
                                   rotate_frac=1, aniso=1, translate_frac=1)

    batch = cfg.batch_size or spec["batch"]
    duration = cfg.duration_mimg if cfg.duration_mimg is not None else spec["duration_mimg"]
    total_steps = cfg.max_steps or round(duration * 1e6 / batch)
    print(f"   [{tag}] batch={batch} duration={duration} MIMG -> {total_steps} steps "
          f"(augment={'on' if augment_on else 'off'})")

    loader = DataLoader(train_dataset, batch_size=batch, shuffle=True,
                        num_workers=cfg.num_workers, drop_last=True, pin_memory=True)

    def cycle(dl):
        while True:
            for b in dl:
                yield b
    it = cycle(loader)
    ckpt = os.path.join(cfg.out_dir, f"{cfg.dataset}_{tag}_edm_ema.pt")
    state_path = os.path.join(cfg.out_dir, f"{cfg.dataset}_{tag}_train_state.pt")

    start_step, nan_skips = 1, 0
    if cfg.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location=dev)
        net.load_state_dict(st["net"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"])
        start_step = st["step"] + 1; nan_skips = st.get("nan_skips", 0)
        try:
            torch.set_rng_state(st["rng_cpu"])
            if st.get("rng_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(st["rng_cuda"])
        except Exception:
            pass
        print(f"   [{tag}] resumed from step {st['step']}/{total_steps} (nan_skips={nan_skips})")
    if start_step > total_steps:
        print(f"   [{tag}] training already complete; using saved EMA for probing")
        return ema

    def save_state(step):
        if not all(torch.isfinite(p).all() for p in ema.parameters()):
            print(f"   [{tag}] EMA non-finite at step {step}; NOT writing checkpoint")
            return
        blob = {"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                "step": step, "nan_skips": nan_skips, "total_steps": total_steps,
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        torch.save(blob, state_path + ".tmp"); os.replace(state_path + ".tmp", state_path)  # atomic
        torch.save(ema.state_dict(), ckpt + ".tmp"); os.replace(ckpt + ".tmp", ckpt)         # EMA artifact

    for step in range(start_step, total_steps + 1):
        x01, _ = next(it)
        images = (x01.to(dev) * 2.0 - 1.0)                    # [0,1] -> [-1,1]
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(net, images, augment_pipe=augment_pipe)
        loss = loss.sum().mul(1.0 / batch)                    # EDM loss scaling

        if not torch.isfinite(loss):
            nan_skips += 1
            if nan_skips == 1 or nan_skips % 200 == 0:
                print(f"   [{tag}] non-finite loss at step {step} -> skipped (total {nan_skips}). "
                      f"fp16 overflow likely; EDM trained CIFAR in fp32 -- consider dropping --fp16.")
            if nan_skips > 2000:
                print(f"   [{tag}] aborting: too many non-finite steps. Re-run without --fp16.")
                break
            continue
        loss.backward()
        if cfg.fp16:                                          # fp16 grad-overflow guard
            gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1e6)
            if not torch.isfinite(gnorm):
                nan_skips += 1
                opt.zero_grad(set_to_none=True)
                if nan_skips == 1 or nan_skips % 200 == 0:
                    print(f"   [{tag}] non-finite grad at step {step} -> skipped (total {nan_skips}).")
                if nan_skips > 2000:
                    print(f"   [{tag}] aborting: too many non-finite steps. Re-run without --fp16.")
                    break
                continue

        cur_nimg = step * batch
        for g in opt.param_groups:                            # EDM lr rampup
            g["lr"] = spec["lr"] * min(cur_nimg / max(cfg.lr_rampup_kimg * 1000, 1e-8), 1.0)
        opt.step()
        with torch.no_grad():                                 # EDM EMA schedule
            hl = cfg.ema_halflife_kimg * 1000
            if cfg.ema_rampup_ratio is not None:
                hl = min(hl, cur_nimg * cfg.ema_rampup_ratio)
            beta = 0.5 ** (batch / max(hl, 1e-8))
            for pe, pm in zip(ema.parameters(), net.parameters()):
                pe.copy_(pm.detach().lerp(pe, beta))
            for be, bm in zip(ema.buffers(), net.buffers()):
                be.copy_(bm)
        if step % cfg.log_every == 0:
            print(f"   [{tag}] step {step}/{total_steps}  loss {loss.item():.4f}")
        if step % cfg.ckpt_every == 0 or step == total_steps:
            save_state(step)
    print(f"   [{tag}] saved EMA -> {ckpt}")
    return ema

@torch.no_grad()
def extract_features_edm(cfg, ema_net, taps, loader, sigma, layers):
    ema_net.eval()
    dev = cfg.device
    feats = {l: [] for l in layers}
    ys = []
    for x01, y in loader:
        x = x01.to(dev) * 2.0 - 1.0
        xt = x + torch.randn_like(x) * sigma                  # perturb clean input to level sigma
        s = torch.full((x.shape[0],), float(sigma), device=dev)
        taps.clear()
        ema_net(xt, s)                                        # precond internal; hooks capture decoder acts
        for l in layers:
            feats[l].append(taps.feats[l].mean(dim=(2, 3)).float().cpu())   # GAP
        ys.append(y)
    return {l: torch.cat(v) for l, v in feats.items()}, torch.cat(ys)


def run_probe_grid_edm(cfg, ema_net, ptr_loader, pte_loader, num_classes, img_size):
    taps = FeatureTaps(ema_net)
    with torch.no_grad():                                     # how many decoder taps?
        d = torch.zeros(2, 3, img_size, img_size, device=cfg.device)
        taps.clear(); ema_net(d, torch.full((2,), 0.1, device=cfg.device))
        n_taps = len(taps.feats)
    layers = cfg.probe_layers or tuple(range(n_taps))
    results = {}
    for sigma in cfg.probe_sigmas:
        Ftr, ytr = extract_features_edm(cfg, ema_net, taps, ptr_loader, sigma, layers)
        Fte, yte = extract_features_edm(cfg, ema_net, taps, pte_loader, sigma, layers)
        for l in layers:
            acc = linear_probe(cfg, Ftr[l], ytr, Fte[l], yte, num_classes)
            results[(l, sigma)] = acc
            print(f"      probe layer={l:2d} ({taps.names[l]:>14s}) sigma={sigma:.2f}  acc={acc*100:.2f}")
    taps.remove()
    best = max(results, key=results.get)
    return {"grid": {f"L{l}_s{s}": a for (l, s), a in results.items()},
            "best": {"layer": best[0], "sigma": best[1], "acc": results[best]},
            "n_taps": n_taps}


def compute_summary(per_condition, dataset):
    summary = {"dataset": dataset, "best_acc": {}, "delta_best": {},
               "delta_fixed_protocol": {}, "clean_protocol": None}
    clean = per_condition.get("clean")
    for tag, r in per_condition.items():
        summary["best_acc"][tag] = r["best"]["acc"]
    if clean is not None:
        cl, cs = clean["best"]["layer"], clean["best"]["sigma"]
        key = f"L{cl}_s{cs}"
        summary["clean_protocol"] = {"layer": cl, "sigma": cs}
        clean_best, clean_fixed = clean["best"]["acc"], clean["grid"][key]
        for tag, r in per_condition.items():
            summary["delta_best"][tag] = clean_best - r["best"]["acc"]
            summary["delta_fixed_protocol"][tag] = clean_fixed - r["grid"].get(key, float("nan"))
    return summary


def save_results(path, per_condition, dataset):
    with open(path, "w") as f:
        json.dump({"per_condition": per_condition,
                   "summary": compute_summary(per_condition, dataset)}, f, indent=2)


def enable_fast_math():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True            # fixed 32x32/64x64 shapes -> autotune
        try:
            torch.set_float32_matmul_precision("high")   # TF32 for fp32 matmuls
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "tiny_imagenet"])
    ap.add_argument("--edm_repo", default="./edm")
    ap.add_argument("--params_json", default="./e1_artifacts/e1_noise_params.json")
    ap.add_argument("--out_dir", default="./e1_runs_edm")
    ap.add_argument("--max_steps", type=int, default=None, help="cap steps (debug)")
    ap.add_argument("--duration_mimg", type=float, default=None, help="override EDM duration")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--no_resume", action="store_true", help="ignore existing train-state; start fresh")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", type=int, default=1, help="noise magnitude level for noisy conditions")
    ap.add_argument("--conditions", default="clean,gaussian,poisson_gaussian,speckle",
                    help="comma-separated conditions to run/bank (resume-aware)")
    args = ap.parse_args()

    cfg = EDMConfig(dataset=args.dataset, edm_repo=args.edm_repo, params_json=args.params_json,
                    out_dir=args.out_dir, max_steps=args.max_steps,
                    duration_mimg=args.duration_mimg, batch_size=args.batch_size,
                    fp16=args.fp16, seed=args.seed, resume=not args.no_resume)
    enable_fast_math()
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | TF32+cuDNN-autotune enabled "
              f"| fp16={cfg.fp16}")
    os.makedirs(cfg.out_dir, exist_ok=True)
    spec = edm_spec(cfg.dataset)
    img_size, img_ch = IMG_SIZE[cfg.dataset], 3

    with open(cfg.params_json) as f:
        table = json.load(f)

    base = E1.Config()
    ptr_ds = E1.make_noisy_dataset(base, table, cfg.dataset, "train", "clean")
    pte_ds = E1.make_noisy_dataset(base, table, cfg.dataset, "test", "clean")
    if cfg.probe_feat_subset > 0:
        ptr_ds = Subset(ptr_ds, torch.randperm(len(ptr_ds))[:cfg.probe_feat_subset].tolist())
    ptr = DataLoader(ptr_ds, batch_size=cfg.probe_batch, num_workers=cfg.num_workers)
    pte = DataLoader(pte_ds, batch_size=cfg.probe_batch, num_workers=cfg.num_workers)

    requested = [c.strip() for c in args.conditions.split(",") if c.strip()]
    conditions = [(c, 0 if c == "clean" else args.level) for c in requested]
    results_path = os.path.join(cfg.out_dir, f"e1_results_edm_{cfg.dataset}.json")

    per_condition = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            per_condition = json.load(f).get("per_condition", {})
        if per_condition:
            print(f"resuming: found completed {sorted(per_condition)} in {results_path}")

    for condition, level in conditions:
        tag = condition if condition == "clean" else f"{condition}_L{level}"
        if tag in per_condition:
            print(f"\n==== condition {tag}: already done, skipping ====")
            continue
        print(f"\n================  condition: {tag}  ================")
        train_ds = E1.make_noisy_dataset(base, table, cfg.dataset, "train", condition, level)
        ema = train_edm(cfg, spec, train_ds, img_ch, img_size, tag)
        per_condition[tag] = run_probe_grid_edm(cfg, ema, ptr, pte,
                                                NUM_CLASSES[cfg.dataset], img_size)
        save_results(results_path, per_condition, cfg.dataset)
        print(f"   [{tag}] results saved -> {results_path}")
        for stale in (os.path.join(cfg.out_dir, f"{cfg.dataset}_{tag}_train_state.pt"),):
            if os.path.exists(stale):
                os.remove(stale)

    print("\n==== E1 (EDM-exact) summary ====")
    print(json.dumps(compute_summary(per_condition, cfg.dataset), indent=2))
    print(f"saved -> {results_path}")


if __name__ == "__main__":
    main()
