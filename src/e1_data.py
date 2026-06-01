import os
import json
import math
import zipfile
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

@dataclass
class Config:
    data_root: str = "./data"              
    out_dir: str = "./e1_artifacts"          
    datasets: List[str] = field(default_factory=lambda: ["cifar10", "tiny_imagenet"])
    families: List[str] = field(default_factory=lambda: ["gaussian", "poisson_gaussian", "speckle"])
    # target average MSE per magnitude level; PSNR = -10*log10(MSE).
    # 0.01/0.04/0.09  ->  20.0 / 14.0 / 10.5 dB
    target_mse: List[float] = field(default_factory=lambda: [0.01, 0.04, 0.09])
    sigma_g: float = 0.02                    
    base_seed: int = 1234                    
    calib_subset: int = 2048                 
    calib_K_search: int = 4                  
    calib_K_report: int = 16                 
    materialize: bool = False                
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def apply_noise(family: str, x: torch.Tensor, s: float, sigma_g: float,
                generator: torch.Generator) -> torch.Tensor:
    if family == "gaussian":
        eps = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        return (x + s * eps).clamp_(0.0, 1.0)
    if family == "poisson_gaussian":
        lam = 1.0 / max(s, 1e-8)
        shot = torch.poisson(x * lam, generator=generator) / lam
        eps = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        return (shot + sigma_g * eps).clamp_(0.0, 1.0)
    if family == "speckle":
        eps = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        return (x * (1.0 + s * eps)).clamp_(0.0, 1.0)
    raise ValueError(f"unknown noise family: {family}")


def strength_to_params(family: str, s: float, sigma_g: float) -> Dict[str, float]:
    if family == "gaussian":
        return {"sigma": s}
    if family == "poisson_gaussian":
        return {"lambda": 1.0 / s, "sigma_g": sigma_g}
    if family == "speckle":
        return {"sigma": s}
    raise ValueError(family)


def init_strength(family: str, V: float, mu: float, m2: float, sigma_g: float) -> float:
    if family == "gaussian":
        return math.sqrt(V)
    if family == "poisson_gaussian":
        lam = mu / max(V - sigma_g ** 2, 1e-6)
        return 1.0 / lam
    if family == "speckle":
        return math.sqrt(V / m2)
    raise ValueError(family)


def _realized_mse(family, x, s, sigma_g, K, seed=999):
    g = torch.Generator(device=x.device).manual_seed(seed)
    xb = x.unsqueeze(0).expand(K, *x.shape)            # (K, N, C, H, W)
    noisy = apply_noise(family, xb, s, sigma_g, g)
    return ((noisy - x.unsqueeze(0)) ** 2).mean().item()


def calibrate(family, V, x, mu, m2, sigma_g, K=4, iters=24):
    s0 = init_strength(family, V, mu, m2, sigma_g)
    lo, hi = s0 * 0.25, s0 * 4.0
    for _ in range(10):                                
        if _realized_mse(family, x, lo, sigma_g, K) <= V:
            break
        lo *= 0.5
    for _ in range(10):
        if _realized_mse(family, x, hi, sigma_g, K) >= V:
            break
        hi *= 2.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _realized_mse(family, x, mid, sigma_g, K) < V:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def variance_profile(family, x, s, sigma_g, nbins=10, K=24, seed=11):
    g = torch.Generator(device=x.device).manual_seed(seed)
    xb = x.unsqueeze(0).expand(K, *x.shape)
    draws = apply_noise(family, xb, s, sigma_g, g)
    var = draws.var(dim=0, unbiased=True).flatten()
    xf = x.flatten()
    edges = torch.linspace(0, 1, nbins + 1, device=x.device)
    centers, profile = [], []
    for b in range(nbins):
        hi_ok = (xf <= edges[b + 1]) if b == nbins - 1 else (xf < edges[b + 1])
        msk = (xf >= edges[b]) & hi_ok
        centers.append(float((edges[b] + edges[b + 1]) / 2))
        profile.append(float(var[msk].mean()) if bool(msk.any()) else float("nan"))
    return centers, profile


def load_cifar10(root):
    from torchvision.datasets import CIFAR10
    out = {}
    for split, train in [("train", True), ("test", False)]:
        ds = CIFAR10(root, train=train, download=True)
        imgs = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()   # uint8 [N,3,32,32]
        labels = torch.tensor(ds.targets, dtype=torch.long)
        out[split] = (imgs, labels)
    return out


_TINY_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def _download_tiny(root):
    zip_path = os.path.join(root, "tiny-imagenet-200.zip")
    extracted = os.path.join(root, "tiny-imagenet-200")
    if os.path.isdir(extracted):
        return extracted
    os.makedirs(root, exist_ok=True)
    if not os.path.isfile(zip_path):
        print(f"[tiny-imagenet] downloading {_TINY_URL} ...")
        urllib.request.urlretrieve(_TINY_URL, zip_path)   # ~240 MB
    print("[tiny-imagenet] extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    return extracted


def load_tiny_imagenet(root):
    from PIL import Image
    base = _download_tiny(root)
    cache = os.path.join(root, "tiny_imagenet_cache.pt")
    if os.path.isfile(cache):
        return torch.load(cache)

    with open(os.path.join(base, "wnids.txt")) as f:
        wnids = sorted(line.strip() for line in f if line.strip())
    wnid_to_idx = {w: i for i, w in enumerate(wnids)}

    def _to_tensor(path):
        return torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1)

    # train
    tr_imgs, tr_lbls = [], []
    for w in wnids:
        img_dir = os.path.join(base, "train", w, "images")
        for fn in sorted(os.listdir(img_dir)):
            tr_imgs.append(_to_tensor(os.path.join(img_dir, fn)))
            tr_lbls.append(wnid_to_idx[w])
    # val (labeled test)
    val_map = {}
    with open(os.path.join(base, "val", "val_annotations.txt")) as f:
        for line in f:
            parts = line.split("\t")
            val_map[parts[0]] = parts[1]
    va_imgs, va_lbls = [], []
    val_dir = os.path.join(base, "val", "images")
    for fn in sorted(os.listdir(val_dir)):
        va_imgs.append(_to_tensor(os.path.join(val_dir, fn)))
        va_lbls.append(wnid_to_idx[val_map[fn]])

    out = {
        "train": (torch.stack(tr_imgs).contiguous(), torch.tensor(tr_lbls, dtype=torch.long)),
        "test":  (torch.stack(va_imgs).contiguous(), torch.tensor(va_lbls, dtype=torch.long)),
    }
    torch.save(out, cache)
    return out


def load_dataset(name, root):
    if name == "cifar10":
        return load_cifar10(root)
    if name == "tiny_imagenet":
        return load_tiny_imagenet(root)
    raise ValueError(name)


_FAMILY_ID = {"clean": 0, "gaussian": 1, "poisson_gaussian": 2, "speckle": 3}


class NoisyDataset(Dataset):
    def __init__(self, images_uint8, labels, family, strength, sigma_g,
                 base_seed, level_index=0, transform=None):
        self.images = images_uint8
        self.labels = labels
        self.family = family
        self.strength = float(strength)
        self.sigma_g = float(sigma_g)
        self.base_seed = int(base_seed)
        self.level_index = int(level_index)
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def _seed_for(self, index):
        fid = _FAMILY_ID[self.family]
        # large coprime-ish multipliers to decorrelate the streams
        s = (self.base_seed * 1_000_003
             + fid * 100_003
             + self.level_index * 10_007
             + index)
        return s % (2 ** 63 - 1)

    def __getitem__(self, index):
        x = self.images[index].float() / 255.0          # [3,H,W] in [0,1] (CPU)
        if self.family != "clean":
            g = torch.Generator().manual_seed(self._seed_for(index))
            x = apply_noise(self.family, x, self.strength, self.sigma_g, g)
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.labels[index])


def compute_moments(images_uint8):
    x = images_uint8.float() / 255.0
    return float(x.mean()), float((x * x).mean())


def build_calibration(cfg: Config):
    os.makedirs(cfg.out_dir, exist_ok=True)
    table = {"config": {"target_mse": cfg.target_mse, "sigma_g": cfg.sigma_g,
                         "base_seed": cfg.base_seed}, "datasets": {}}

    for ds_name in cfg.datasets:
        print(f"\n================  {ds_name}  ================")
        data = load_dataset(ds_name, cfg.data_root)
        train_imgs, _ = data["train"]
        mu, m2 = compute_moments(train_imgs)
        print(f"moments: mu={mu:.4f}  E[x^2]={m2:.4f}  "
              f"(train {tuple(train_imgs.shape)}, test {tuple(data['test'][0].shape)})")

        idx = torch.randperm(train_imgs.shape[0])[: cfg.calib_subset]
        xcal = (train_imgs[idx].float() / 255.0).to(cfg.device)

        ds_rec = {"moments": {"mu": mu, "m2": m2},
                  "shape": list(train_imgs.shape[1:]),
                  "n_train": int(train_imgs.shape[0]),
                  "n_test": int(data["test"][0].shape[0]),
                  "levels": {}}

        for li, V in enumerate(cfg.target_mse):
            psnr = -10 * math.log10(V)
            print(f"\n  level {li}: target MSE={V:.4f}  (PSNR={psnr:.2f} dB)")
            ds_rec["levels"][f"L{li}"] = {"target_mse": V, "target_psnr": psnr, "families": {}}
            for fam in cfg.families:
                s = calibrate(fam, V, xcal, mu, m2, cfg.sigma_g, K=cfg.calib_K_search)
                mse = _realized_mse(fam, xcal, s, cfg.sigma_g, K=cfg.calib_K_report)
                rec = {
                    "strength": s,
                    "params": strength_to_params(fam, s, cfg.sigma_g),
                    "realized_mse": mse,
                    "realized_psnr": -10 * math.log10(max(mse, 1e-9)),
                }
                ds_rec["levels"][f"L{li}"]["families"][fam] = rec
                pp = (f"lambda={rec['params']['lambda']:8.2f}"
                      if fam == "poisson_gaussian" else f"sigma={s:6.3f}      ")
                print(f"     {fam:18s} {pp}  realizedMSE={mse:.4f}  "
                      f"PSNR={rec['realized_psnr']:.2f}")

        table["datasets"][ds_name] = ds_rec
        _plot_signatures(cfg, ds_name, xcal, ds_rec)

    params_path = os.path.join(cfg.out_dir, "e1_noise_params.json")
    with open(params_path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\nSaved calibration table -> {params_path}")
    return table


def _plot_signatures(cfg, ds_name, xcal, ds_rec):
    levels = list(ds_rec["levels"].keys())
    fig, axes = plt.subplots(1, len(levels), figsize=(4.2 * len(levels), 3.4), squeeze=False)
    for j, lk in enumerate(levels):
        ax = axes[0][j]
        for fam in cfg.families:
            s = ds_rec["levels"][lk]["families"][fam]["strength"]
            centers, prof = variance_profile(fam, xcal, s, cfg.sigma_g)
            ax.plot(centers, prof, marker="o", ms=3, label=fam)
        V = ds_rec["levels"][lk]["target_mse"]
        ax.set_title(f"{ds_name}  {lk}  (PSNR={-10*math.log10(V):.1f} dB)")
        ax.set_xlabel("pixel intensity"); ax.set_ylabel("noise variance")
        if j == 0:
            ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(cfg.out_dir, f"noise_signatures_{ds_name}.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  signature figure -> {path}")


def make_noisy_dataset(cfg, table, ds_name, split, condition, level_index=0):
    data = load_dataset(ds_name, cfg.data_root)
    imgs, labels = data[split]
    if condition == "clean":
        return NoisyDataset(imgs, labels, "clean", 0.0, cfg.sigma_g, cfg.base_seed)
    rec = table["datasets"][ds_name]["levels"][f"L{level_index}"]["families"][condition]
    return NoisyDataset(imgs, labels, condition, rec["strength"], cfg.sigma_g,
                        cfg.base_seed, level_index=level_index)


def materialize(cfg, table):
    mat_dir = os.path.join(cfg.out_dir, "materialized")
    os.makedirs(mat_dir, exist_ok=True)
    for ds_name in cfg.datasets:
        for split in ["train", "test"]:
            for li in range(len(cfg.target_mse)):
                for cond in ["clean"] + cfg.families:
                    if cond == "clean" and li > 0:
                        continue                          # clean has no levels
                    dset = make_noisy_dataset(cfg, table, ds_name, split, cond, li)
                    buf = torch.empty((len(dset), *dset.images.shape[1:]), dtype=torch.uint8)
                    lbl = torch.empty(len(dset), dtype=torch.long)
                    loader = DataLoader(dset, batch_size=512, num_workers=4)
                    k = 0
                    for xb, yb in loader:
                        n = xb.shape[0]
                        buf[k:k + n] = (xb * 255.0).round().clamp(0, 255).to(torch.uint8)
                        lbl[k:k + n] = yb
                        k += n
                    tag = "clean" if cond == "clean" else f"{cond}_L{li}"
                    out = os.path.join(mat_dir, f"{ds_name}_{split}_{tag}.pt")
                    torch.save({"images": buf, "labels": lbl}, out)
                    print(f"  wrote {out}  {tuple(buf.shape)}")


def sanity_check(cfg, table):
    ds_name = cfg.datasets[0]
    dset = make_noisy_dataset(cfg, table, ds_name, "train", "speckle", level_index=1)
    clean = make_noisy_dataset(cfg, table, ds_name, "train", "clean")
    x1a, _ = dset[0]; x1b, _ = dset[0]
    assert torch.allclose(x1a, x1b), "noise is not deterministic per image!"
    # realized MSE over a small batch, clean vs speckle-L1
    mses = []
    for i in range(256):
        xc, _ = clean[i]; xn, _ = dset[i]
        mses.append(((xn - xc) ** 2).mean().item())
    target = table["datasets"][ds_name]["levels"]["L1"]["target_mse"]
    print(f"\nsanity[{ds_name} speckle L1]: determinism OK"
          f"\ndataset-path MSE={np.mean(mses):.4f} vs target {target:.4f}")


def main():
    cfg = Config()
    print(f"device={cfg.device}  datasets={cfg.datasets}  "
          f"levels(MSE)={cfg.target_mse}  families={cfg.families}")
    table = build_calibration(cfg)
    sanity_check(cfg, table)
    if cfg.materialize:
        print("\nMaterializing noisy datasets to disk (this is large) ...")
        materialize(cfg, table)
    print("\nDone")


if __name__ == "__main__":
    main()
