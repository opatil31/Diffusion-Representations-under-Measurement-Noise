import os
import math
import copy
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import e1_data as E1

@dataclass
class Config:
    dataset: str = "cifar10"                 
    out_dir: str = "./e1_runs"
    conditions: List[tuple] = field(default_factory=lambda: [
        ("clean", 0),
        ("gaussian", 1), ("poisson_gaussian", 1), ("speckle", 1),
    ])
    params_json: str = "./e1_artifacts/e1_noise_params.json"

    base_ch: int = 128
    ch_mult: tuple = (2, 2, 2)
    num_res_blocks: int = 4
    attn_res: tuple = (16,)
    dropout: float = 0.13
    sigma_data: float = 0.5                   
    p_mean: float = -1.2
    p_std: float = 1.2

    # --- training ---
    train_steps: int = 50000                  
    batch_size: int = 256                     
    lr: float = 1e-3                           
    warmup: int = 2000                         
    ema_halflife_kimg: float = 500.0           
    ema_rampup_ratio: float = 0.05
    num_workers: int = 4
    log_every: int = 500
    ckpt_every: int = 5000

    probe_sigmas: tuple = (0.0, 0.05, 0.1, 0.2)   
    probe_layers: Optional[tuple] = None          
    probe_epochs: int = 60
    probe_lr: float = 1e-3
    probe_batch: int = 256
    probe_feat_subset: int = 0                    

    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_ch, dropout):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb = nn.Linear(temb_ch, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, C, H * W).unbind(1)
        attn = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(C), dim=-1)
        h = (v @ attn.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.proj(h)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__(); self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x): return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__(); self.op = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x): return self.op(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """Compact DDPM-style UNet. Collects up-block activations in self.up_feats."""
    def __init__(self, img_ch, img_size, base_ch, ch_mult, num_res_blocks, attn_res, dropout):
        super().__init__()
        temb_ch = base_ch * 4
        self.temb_mlp = nn.Sequential(nn.Linear(base_ch, temb_ch), nn.SiLU(),
                                      nn.Linear(temb_ch, temb_ch))
        self.base_ch = base_ch
        self.conv_in = nn.Conv2d(img_ch, base_ch, 3, padding=1)

        # ---- down path ----
        self.down = nn.ModuleList()
        chs = [base_ch]
        ch = base_ch
        res = img_size
        for i, mult in enumerate(ch_mult):
            out_ch = base_ch * mult
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(ch, out_ch, temb_ch, dropout)])
                ch = out_ch
                if res in attn_res:
                    block.append(AttnBlock(ch))
                self.down.append(block); chs.append(ch)
            if i != len(ch_mult) - 1:
                self.down.append(nn.ModuleList([Downsample(ch)])); chs.append(ch); res //= 2

        # ---- middle ----
        self.mid = nn.ModuleList([ResBlock(ch, ch, temb_ch, dropout),
                                  AttnBlock(ch),
                                  ResBlock(ch, ch, temb_ch, dropout)])

        # ---- up path ----
        self.up = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = base_ch * mult
            for _ in range(num_res_blocks + 1):
                block = nn.ModuleList([ResBlock(ch + chs.pop(), out_ch, temb_ch, dropout)])
                ch = out_ch
                if res in attn_res:
                    block.append(AttnBlock(ch))
                self.up.append(block)
            if i != 0:
                self.up.append(nn.ModuleList([Upsample(ch)])); res *= 2

        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, img_ch, 3, padding=1)
        self.up_feats: List[torch.Tensor] = []

    def forward(self, x, c_noise, collect=False):
        temb = self.temb_mlp(timestep_embedding(c_noise, self.base_ch))
        h = self.conv_in(x)
        hs = [h]
        for block in self.down:
            for layer in block:
                h = layer(h, temb) if isinstance(layer, ResBlock) else layer(h)
            hs.append(h)
        for layer in self.mid:
            h = layer(h, temb) if isinstance(layer, ResBlock) else layer(h)
        self.up_feats = []
        for block in self.up:
            for layer in block:
                if isinstance(layer, ResBlock):
                    h = layer(torch.cat([h, hs.pop()], dim=1), temb)
                elif isinstance(layer, Upsample):
                    h = layer(h)
                else:
                    h = layer(h)
            if collect:
                self.up_feats.append(h)
        return self.conv_out(F.silu(self.norm_out(h)))


class EDM(nn.Module):
    def __init__(self, unet, sigma_data, p_mean, p_std):
        super().__init__()
        self.unet = unet
        self.sigma_data = sigma_data
        self.p_mean, self.p_std = p_mean, p_std

    def precond(self, x, sigma, collect=False):
        sd = self.sigma_data
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in = 1.0 / (sigma ** 2 + sd ** 2).sqrt()
        c_noise = 0.25 * sigma.log()
        v = c_in.view(-1, 1, 1, 1)
        out = self.unet(v * x, c_noise.flatten(), collect=collect)
        return c_skip.view(-1, 1, 1, 1) * x + c_out.view(-1, 1, 1, 1) * out

    def loss(self, x0):                      # x0 in [-1,1]
        B = x0.shape[0]
        sigma = (self.p_mean + self.p_std * torch.randn(B, device=x0.device)).exp()
        s = sigma.view(-1, 1, 1, 1)
        n = torch.randn_like(x0) * s
        D = self.precond(x0 + n, sigma)
        w = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        return (w.view(-1, 1, 1, 1) * (D - x0) ** 2).mean()


def to_model_space(x01):     # [0,1] -> [-1,1]
    return x01 * 2.0 - 1.0


def train_diffusion(cfg, train_dataset, img_ch, img_size, tag):
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    unet = UNet(img_ch, img_size, cfg.base_ch, cfg.ch_mult,
                cfg.num_res_blocks, cfg.attn_res, cfg.dropout).to(cfg.device)
    model = EDM(unet, cfg.sigma_data, cfg.p_mean, cfg.p_std).to(cfg.device)
    ema = copy.deepcopy(unet).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, drop_last=True, pin_memory=True)

    def cycle(dl):
        while True:
            for b in dl:
                yield b
    it = cycle(loader)

    ckpt_path = os.path.join(cfg.out_dir, f"{cfg.dataset}_{tag}_unet_ema.pt")
    model.train()
    for step in range(1, cfg.train_steps + 1):
        x01, _ = next(it)
        x0 = to_model_space(x01.to(cfg.device))
        loss = model.loss(x0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for g in opt.param_groups:                      # linear warmup
            g["lr"] = cfg.lr * min(1.0, step / cfg.warmup)
        opt.step()
        with torch.no_grad():                           # EDM EMA schedule
            cur_nimg = step * cfg.batch_size
            halflife = cfg.ema_halflife_kimg * 1000
            if cfg.ema_rampup_ratio is not None:
                halflife = min(halflife, cur_nimg * cfg.ema_rampup_ratio)
            beta = 0.5 ** (cfg.batch_size / max(halflife, 1e-8))
            for pe, pm in zip(ema.parameters(), unet.parameters()):
                pe.copy_(pm.detach().lerp(pe, beta))
            for be, bm in zip(ema.buffers(), unet.buffers()):
                be.copy_(bm)
        if step % cfg.log_every == 0:
            print(f"   [{tag}] step {step}/{cfg.train_steps}  loss {loss.item():.4f}")
        if step % cfg.ckpt_every == 0 or step == cfg.train_steps:
            torch.save(ema.state_dict(), ckpt_path)
    print(f"   [{tag}] saved EMA UNet -> {ckpt_path}")
    return ema      # frozen EMA UNet used for probing


@torch.no_grad()
def extract_features(cfg, ema_unet, edm_precond, loader, sigma, layers):
    """Return {layer_idx: pooled_features [N, C]} and labels, at one extraction sigma."""
    ema_unet.eval()
    feats: Dict[int, list] = {l: [] for l in layers}
    ys = []
    sd = cfg.sigma_data
    for x01, y in loader:
        x = to_model_space(x01.to(cfg.device))
        s = torch.full((x.shape[0],), max(sigma, 1e-3), device=cfg.device)
        if sigma > 0:
            x = x + torch.randn_like(x) * s.view(-1, 1, 1, 1)
        # preconditioned forward through the EMA net, collecting up-block activations
        c_in = 1.0 / (s ** 2 + sd ** 2).sqrt()
        c_noise = 0.25 * s.log()
        ema_unet(c_in.view(-1, 1, 1, 1) * x, c_noise, collect=True)
        for l in layers:
            feats[l].append(ema_unet.up_feats[l].mean(dim=(2, 3)).cpu())  # GAP
        ys.append(y)
    feats = {l: torch.cat(v) for l, v in feats.items()}
    return feats, torch.cat(ys)


def linear_probe(cfg, Xtr, ytr, Xte, yte, num_classes):
    dev = cfg.device
    Xtr, Xte = Xtr.to(dev), Xte.to(dev)
    ytr, yte = ytr.to(dev), yte.to(dev)
    bn = nn.BatchNorm1d(Xtr.shape[1], affine=False).to(dev)   # parameter-free, MAE/l-DAE style
    clf = nn.Linear(Xtr.shape[1], num_classes).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=cfg.probe_lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.probe_epochs)
    n = Xtr.shape[0]
    bn.train()
    # fit BN stats once on the (frozen) features
    with torch.no_grad():
        for i in range(0, n, cfg.probe_batch):
            bn(Xtr[i:i + cfg.probe_batch])
    bn.eval()
    Xtr_n = bn(Xtr); Xte_n = bn(Xte)
    for _ in range(cfg.probe_epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, cfg.probe_batch):
            idx = perm[i:i + cfg.probe_batch]
            loss = F.cross_entropy(clf(Xtr_n[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sched.step()
    with torch.no_grad():
        acc = (clf(Xte_n).argmax(1) == yte).float().mean().item()
    return acc


def run_probe_grid(cfg, ema_unet, edm, probe_train_loader, probe_test_loader,
                   num_classes, n_up_layers):
    layers = cfg.probe_layers or tuple(range(n_up_layers))
    results = {}      # (layer, sigma) -> acc
    for sigma in cfg.probe_sigmas:
        Ftr, ytr = extract_features(cfg, ema_unet, edm, probe_train_loader, sigma, layers)
        Fte, yte = extract_features(cfg, ema_unet, edm, probe_test_loader, sigma, layers)
        for l in layers:
            acc = linear_probe(cfg, Ftr[l], ytr, Fte[l], yte, num_classes)
            results[(l, sigma)] = acc
            print(f"      probe layer={l:2d} sigma={sigma:.2f}  acc={acc*100:.2f}")
    best = max(results, key=results.get)
    return {"grid": {f"L{l}_s{s}": a for (l, s), a in results.items()},
            "best": {"layer": best[0], "sigma": best[1], "acc": results[best]}}


NUM_CLASSES = {"cifar10": 10, "tiny_imagenet": 200}
IMG_SIZE = {"cifar10": 32, "tiny_imagenet": 64}


def arch_for(dataset):
    if dataset == "cifar10":
        return dict(ch_mult=(2, 2, 2), num_res_blocks=4, attn_res=(16,), dropout=0.13, lr=1e-3)
    if dataset == "tiny_imagenet":          # 64x64, mirrors EDM FFHQ/AFHQ-64 (cres=1,2,2,2)
        return dict(ch_mult=(1, 2, 2, 2), num_res_blocks=4, attn_res=(16,), dropout=0.10, lr=2e-4)
    raise ValueError(dataset)


def apply_arch(cfg):
    a = arch_for(cfg.dataset)
    cfg.ch_mult = a["ch_mult"]; cfg.num_res_blocks = a["num_res_blocks"]
    cfg.attn_res = a["attn_res"]; cfg.dropout = a["dropout"]; cfg.lr = a["lr"]
    return cfg


def run_condition(cfg, table, condition, level, probe_loaders, img_ch, img_size, tag):
    train_ds = E1.make_noisy_dataset(E1.Config(), table, cfg.dataset, "train",
                                     condition, level)
    ema = train_diffusion(cfg, train_ds, img_ch, img_size, tag)
    edm = EDM(ema, cfg.sigma_data, cfg.p_mean, cfg.p_std)
    # number of up-blocks that produce features
    with torch.no_grad():
        dummy = torch.zeros(2, img_ch, img_size, img_size, device=cfg.device)
        ema(dummy, torch.zeros(2, device=cfg.device), collect=True)
        n_up = len(ema.up_feats)
    res = run_probe_grid(cfg, ema, edm, probe_loaders[0], probe_loaders[1],
                         NUM_CLASSES[cfg.dataset], n_up)
    return res


def main():
    cfg = Config()
    apply_arch(cfg)                          
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(cfg.params_json) as f:
        table = json.load(f)

    img_size = IMG_SIZE[cfg.dataset]; img_ch = 3

    base = E1.Config()
    probe_train = E1.make_noisy_dataset(base, table, cfg.dataset, "train", "clean")
    probe_test = E1.make_noisy_dataset(base, table, cfg.dataset, "test", "clean")
    if cfg.probe_feat_subset > 0:
        idx = torch.randperm(len(probe_train))[: cfg.probe_feat_subset].tolist()
        probe_train = torch.utils.data.Subset(probe_train, idx)
    pl = (DataLoader(probe_train, batch_size=cfg.probe_batch, num_workers=cfg.num_workers),
          DataLoader(probe_test, batch_size=cfg.probe_batch, num_workers=cfg.num_workers))

    all_results = {}
    for condition, level in cfg.conditions:
        tag = condition if condition == "clean" else f"{condition}_L{level}"
        print(f"\n================  condition: {tag}  ================")
        all_results[tag] = run_condition(cfg, table, condition, level, pl,
                                         img_ch, img_size, tag)

    clean_acc = all_results.get("clean", {}).get("best", {}).get("acc")
    summary = {"dataset": cfg.dataset, "best_acc": {}, "delta_vs_clean": {}}
    for tag, r in all_results.items():
        a = r["best"]["acc"]
        summary["best_acc"][tag] = a
        if clean_acc is not None:
            summary["delta_vs_clean"][tag] = clean_acc - a
    out = os.path.join(cfg.out_dir, f"e1_results_{cfg.dataset}.json")
    with open(out, "w") as f:
        json.dump({"per_condition": all_results, "summary": summary}, f, indent=2)
    print("\n==== E1 summary ====")
    print(json.dumps(summary, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
