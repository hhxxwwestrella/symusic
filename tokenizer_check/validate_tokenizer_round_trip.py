#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Round-trip sanity check for token shards:
  shards (ids/attn) -> Aria backbone -> mean pool -> proj heads -> cosine stats.

This matches the loading/pooling conventions in phrase2context_clip.py.
"""

import argparse, io, os, random, tarfile, json, math, sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

# ---------- dynamic import of your training script ----------
import importlib.util
def load_module_from_path(py_path: Path):
    spec = importlib.util.spec_from_file_location(py_path.stem, str(py_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[py_path.stem] = mod
    spec.loader.exec_module(mod)
    return mod

# ---------- shard reader ----------
def list_shards(shards_dir: Path) -> List[Path]:
    out = []
    for p in shards_dir.iterdir():
        if p.is_file() and p.name.startswith("tokens-") and (".tar" in p.name):
            out.append(p)
    return sorted(out)

def collect_items(shard_paths: List[Path], max_items: int) -> dict:
    rng = random.Random(12345)
    picked = []
    if not shard_paths:
        raise FileNotFoundError("No shard files found in --shards-dir")
    per_shard = max(1, math.ceil(max_items / len(shard_paths)))
    for sp in shard_paths:
        if len(picked) >= max_items: break
        with tarfile.open(sp, "r:*") as tf:
            metas = [m for m in tf.getmembers() if m.name.endswith(".meta.json")]
            if not metas: continue
            rng.shuffle(metas)
            for m in metas[:per_shard]:
                if len(picked) >= max_items: break
                base = m.name[:-10]  # strip .meta.json
                def get(name):
                    mem = tf.extractfile(name)
                    if mem is None: return None
                    return mem.read()
                meta = json.load(io.BytesIO(get(m.name)))
                pid = base + ".phrase_ids.npy"
                pat = base + ".phrase_attn.npy"
                cid = base + ".context_ids.npy"
                cat = base + ".context_attn.npy"
                phr_ids = np.load(io.BytesIO(get(pid)))
                phr_attn = np.load(io.BytesIO(get(pat)))
                ctx_ids = np.load(io.BytesIO(get(cid)))
                ctx_attn = np.load(io.BytesIO(get(cat)))
                if min(len(phr_ids), len(ctx_ids)) == 0:
                    continue
                picked.append((phr_ids, phr_attn, ctx_ids, ctx_attn, meta))
    if not picked:
        raise RuntimeError("No usable items found in shards (all empty?).")
    return {
        "phrase_ids":  [torch.from_numpy(x[0]).long() for x in picked],
        "phrase_attn": [torch.from_numpy(x[1]).long() for x in picked],
        "context_ids":  [torch.from_numpy(x[2]).long() for x in picked],
        "context_attn": [torch.from_numpy(x[3]).long() for x in picked],
        "meta": [x[4] for x in picked],
    }

def pad_batch(seqs: List[torch.Tensor], pad_id: int = 0) -> torch.Tensor:
    if not seqs: return torch.empty(0, 0, dtype=torch.long)
    L = max(s.numel() for s in seqs)
    B = len(seqs)
    out = torch.full((B, L), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :s.numel()] = s
    return out

# ---------- pooling & stats ----------
def mean_pool(seq_emb: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    """
    seq_emb: [B, L, D] or [B, D]
    attn:    [B, L] (0/1)
    Returns pooled [B, D].
    """
    if seq_emb.dim() == 2:
        return seq_emb
    m = attn.unsqueeze(-1).float()
    denom = m.sum(dim=1).clamp(min=1e-6)
    return (seq_emb * m).sum(dim=1) / denom

def cosine_stats(zp: torch.Tensor, zc: torch.Tensor):
    sim = zp @ zc.T  # [B, B]
    pos = sim.diag()
    # hardest neg per row
    B = sim.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=sim.device)
    hard_neg = sim.masked_fill(eye, -1e9).max(dim=1).values
    r1 = (sim.argmax(dim=1) == torch.arange(B, device=sim.device)).float().mean()
    return {
        "mean_pos": pos.mean().item(),
        "mean_hard_neg": hard_neg.mean().item(),
        "margin": (pos - hard_neg).mean().item(),
        "r1": r1.item(),
        "sim": sim,
    }

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Round-trip validator for token shards (ids/attn)")
    ap.add_argument("--shards-dir", type=Path, required=True)
    ap.add_argument("--script-path", type=Path, default=Path("phrase2context_clip.py"))
    ap.add_argument("--aria-ckpt", type=Path, required=True)
    ap.add_argument("--proj-ckpt", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    # import your training script to reuse config/model helpers
    mod = load_module_from_path(args.script_path)

    # Build Aria backbone exactly like your encode/query paths do
    from aria.model import TransformerEMB, ModelConfig
    from aria.config import load_model_config
    from ariautils.tokenizer import AbsTokenizer

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_cfg = ModelConfig(**load_model_config(name="medium-emb"))
    model_cfg.set_vocab_size(AbsTokenizer().vocab_size)
    aria = TransformerEMB(model_cfg)
    sd = load_file(filename=str(args.aria_ckpt))
    aria.load_state_dict(sd, strict=True)
    aria.to(device).eval()

    # Rebuild BOTH heads from checkpoint saved by your train()
    ckpt = torch.load(str(args.proj_ckpt), map_location="cpu")

    def build_head(sd_head: dict) -> torch.nn.Module:
        w1 = sd_head["0.weight"]; w2 = sd_head["2.weight"]
        base_dim = w1.shape[1]; hid = w1.shape[0]; proj_dim = w2.shape[0]
        head = torch.nn.Sequential(
            torch.nn.Linear(base_dim, hid),
            torch.nn.ReLU(),
            torch.nn.Linear(hid, proj_dim),
        ).to(device).eval()
        head.load_state_dict(sd_head)
        return head

    phrase_head = build_head(ckpt["phrase_head"]).eval()
    context_head = build_head(ckpt["context_head"]).eval()

    # Use your robust _aria_forward if present; else provide a local equivalent
    if hasattr(mod, "_aria_forward"):
        def aria_forward(x_ids, x_mask):
            return mod._aria_forward(aria, x_ids, x_mask)
    else:
        def aria_forward(m, input_ids, attention_mask):
            # Try (ids,mask) then (ids) then (x=,mask=)
            try:
                out = m(input_ids, attention_mask)
                return out[0] if isinstance(out, (tuple, list)) else out
            except TypeError:
                pass
            try:
                out = m(input_ids)
                return out[0] if isinstance(out, (tuple, list)) else out
            except TypeError:
                pass
            out = m(x=input_ids, mask=attention_mask)  # may still raise TypeError
            return out[0] if isinstance(out, (tuple, list)) else out
        def aria_forward(x_ids, x_mask):  # bind m
            return aria_forward.__wrapped__(aria, x_ids, x_mask)

    # Sample and pad items
    shards = list_shards(args.shards_dir)[:2]
    data = collect_items(shards, args.samples)
    B = len(data["meta"])
    assert B >= 2, "Need at least 2 items."

    phrase_ids  = pad_batch(data["phrase_ids"])
    phrase_attn = pad_batch(data["phrase_attn"])
    context_ids  = pad_batch(data["context_ids"])
    context_attn = pad_batch(data["context_attn"])

    # Batch through model
    all_zp, all_zc = [], []
    with torch.no_grad():
        for start in range(0, B, args.batch_size):
            end = min(B, start + args.batch_size)
            ids_p = phrase_ids[start:end].to(device, non_blocking=True)
            msk_p = phrase_attn[start:end].to(device, non_blocking=True)
            ids_c = context_ids[start:end].to(device, non_blocking=True)
            msk_c = context_attn[start:end].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                seq_p = aria_forward(ids_p, msk_p)  # [B, L, D] or [B, D]
                seq_c = aria_forward(ids_c, msk_c)
                base_p = mean_pool(seq_p, msk_p)    # [B, D]
                base_c = mean_pool(seq_c, msk_c)

                zp = phrase_head(base_p)
                zc = context_head(base_c)
                zp = F.normalize(zp, p=2, dim=1)
                zc = F.normalize(zc, p=2, dim=1)

            if not torch.isfinite(zp).all() or not torch.isfinite(zc).all():
                raise RuntimeError("Non-finite values encountered in embeddings.")
            all_zp.append(zp)
            all_zc.append(zc)

    zp = torch.cat(all_zp, dim=0)
    zc = torch.cat(all_zc, dim=0)
    stats = cosine_stats(zp, zc)

    print("\n=== Round-trip sanity ===")
    print(f"Samples:           {B}")
    print(f"Mean pos sim:      {stats['mean_pos']:.4f}")
    print(f"Mean hard neg sim: {stats['mean_hard_neg']:.4f}")
    print(f"Avg margin:        {stats['margin']:.4f}")
    print(f"Recall@1:          {stats['r1']*100:.1f}%")

    # Show a couple best/worst rows for quick eyeballing
    sim = stats["sim"].cpu()
    diag = sim.diag()
    top = torch.topk(diag, k=min(3, B)).indices.tolist()
    worst = torch.topk(diag, k=min(3, B), largest=False).indices.tolist()

    def show(i):
        row = sim[i]
        hn = row.masked_fill(torch.eye(B, dtype=torch.bool), -1e9).max().item()
        print(f"- idx {i:3d}: pos={row[i]:.4f}  hard_neg={hn:.4f}  src={data['meta'][i]['src_path']}")

    print("\nTop positives:")
    for i in top: show(i)
    print("\nWorst positives:")
    for i in worst: show(i)

if __name__ == "__main__":
    main()
