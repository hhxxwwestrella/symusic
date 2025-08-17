#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Round-trip sanity check:
  shards (token ids) -> Aria backbone -> projection heads -> cosine similarity stats.
Requires your patched phrase2context_clip.py with the token-batch fast path.
"""

import argparse, io, os, random, tarfile, json, math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---- Import your training script bits ----
# Assumes phrase2context_clip.py is on PYTHONPATH or in the same folder.
import importlib.util, sys

def load_module_from_path(py_path: Path):
    spec = importlib.util.spec_from_file_location(py_path.stem, str(py_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[py_path.stem] = mod
    spec.loader.exec_module(mod)
    return mod

# ------------------ Shard reader ------------------

def list_shards(shards_dir: Path) -> List[Path]:
    return sorted([p for p in shards_dir.iterdir() if p.name.startswith("tokens-") and p.suffix in (".tar", ".gz", ".tgz", ".bz2", ".xz") or p.suffixes[-2:] == [".tar", ".gz"]])

def collect_items(shard_paths: List[Path], max_items: int) -> dict:
    """
    Returns a dict of lists: phrase_ids, phrase_attn, context_ids, context_attn, meta
    """
    rng = random.Random(12345)
    picked = []
    # To keep it simple, walk shards round-robin and sample a few per shard
    per_shard = max(1, math.ceil(max_items / max(1, len(shard_paths))))
    for sp in shard_paths:
        if len(picked) >= max_items: break
        with tarfile.open(sp, "r:*") as tf:
            # meta entries enumerate items
            metas = [m for m in tf.getmembers() if m.name.endswith(".meta.json")]
            if not metas: continue
            rng.shuffle(metas)
            for m in metas[:per_shard]:
                if len(picked) >= max_items: break
                base = m.name[:-10]  # strip ".meta.json"
                def get_npy(name):
                    mem = tf.extractfile(name)
                    if mem is None: return None
                    data = mem.read()
                    return np.load(io.BytesIO(data))
                meta = json.load(io.BytesIO(tf.extractfile(m).read()))
                pid = base + "phrase_ids.npy"
                pat = base + "phrase_attn.npy"
                cid = base + "context_ids.npy"
                cat = base + "context_attn.npy"
                phr_ids = get_npy(pid); phr_attn = get_npy(pat)
                ctx_ids = get_npy(cid); ctx_attn = get_npy(cat)
                if any(x is None for x in (phr_ids, phr_attn, ctx_ids, ctx_attn)): 
                    continue
                # minimal sanity
                if len(phr_ids) == 0 or len(ctx_ids) == 0: 
                    continue
                picked.append((phr_ids, phr_attn, ctx_ids, ctx_attn, meta))
    # pack to dict of tensors (we’ll pad in torch)
    return {
        "phrase_ids": [torch.from_numpy(x[0]).long() for x in picked],
        "phrase_attn": [torch.from_numpy(x[1]).long() for x in picked],
        "context_ids": [torch.from_numpy(x[2]).long() for x in picked],
        "context_attn": [torch.from_numpy(x[3]).long() for x in picked],
        "meta": [x[4] for x in picked],
    }

def pad_batch(seqs: List[torch.Tensor], pad: int = 0) -> torch.Tensor:
    if not seqs: return torch.empty(0, 0, dtype=torch.long)
    max_len = max(s.numel() for s in seqs)
    out = torch.full((len(seqs), max_len), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        L = s.numel()
        out[i, :L] = s
    return out

# ------------------ Validation ------------------

def compute_stats(zp: torch.Tensor, zc: torch.Tensor):
    """zp, zc: [B, D], L2-normalized"""
    sim = zp @ zc.T  # [B, B]
    diag = sim.diag()                       # positives
    # hardest negatives per row (exclude self)
    neg_mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    hard_neg = (sim.masked_fill(~neg_mask, -1e9)).max(dim=1).values
    r1 = (sim.argmax(dim=1) == torch.arange(sim.size(0), device=sim.device)).float().mean()
    return {
        "mean_pos": diag.mean().item(),
        "mean_hard_neg": hard_neg.mean().item(),
        "margin": (diag - hard_neg).mean().item(),
        "recall_at_1": r1.item(),
        "sim_matrix": sim,
    }

def main():
    ap = argparse.ArgumentParser("Round-trip validator for token shards")
    ap.add_argument("--shards-dir", type=Path, required=True)
    ap.add_argument("--script-path", type=Path, default=Path("phrase2context_clip.py"),
                    help="Path to your training script with DualEncoder & token fast path")
    ap.add_argument("--aria-ckpt", type=Path, required=True)
    ap.add_argument("--proj-ckpt", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    mod = load_module_from_path(args.script_path)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model pieces
    aria = mod.load_aria(str(args.aria-ckpt), device=device)   # uses your helper
    model = mod.DualEncoder(aria_model=aria, device=device)    # constructor name per your file
    # Load proj heads
    ckpt = torch.load(str(args.proj-ckpt), map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.eval().to(device)

    # Sample items
    shards = list_shards(args.shards-dir)
    assert shards, f"No shards found in {args.shards_dir}"
    data = collect_items(shards, args.samples)
    B = len(data["meta"])
    assert B > 1, "Need at least 2 items to compute negatives."

    # Pad to tensors
    phrase_ids  = pad_batch(data["phrase_ids"])
    phrase_attn = pad_batch(data["phrase_attn"])
    context_ids  = pad_batch(data["context_ids"])
    context_attn = pad_batch(data["context_attn"])

    # Mini-batch in case samples > batch-size
    all_zp, all_zc = [], []
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    with torch.no_grad():
        for start in range(0, B, args.batch_size):
            end = min(B, start + args.batch_size)
            batch = {
                "phrase_ids":  phrase_ids[start:end].to(device, non_blocking=True),
                "phrase_attn": phrase_attn[start:end].to(device, non_blocking=True),
                "context_ids":  context_ids[start:end].to(device, non_blocking=True),
                "context_attn": context_attn[start:end].to(device, non_blocking=True),
            }
            with torch.cuda.amp.autocast(enabled=args.amp):
                zp, zc = model(batch)  # uses token fast path you added
            # numerical sanity
            for name, z in (("phrase", zp), ("context", zc)):
                if not torch.isfinite(z).all():
                    raise RuntimeError(f"Non-finite values in {name} embeddings")
                # L2-normalize if your model.forward doesn't already
                z = F.normalize(z, p=2, dim=1)
                if name == "phrase": all_zp.append(z)
                else: all_zc.append(z)

    zp = torch.cat(all_zp, dim=0)
    zc = torch.cat(all_zc, dim=0)

    stats = compute_stats(zp, zc)
    print("\n=== Round-trip sanity ===")
    print(f"Samples: {B}")
    print(f"Mean pos sim:     {stats['mean_pos']:.4f}")
    print(f"Mean hard neg sim:{stats['mean_hard_neg']:.4f}")
    print(f"Avg margin:       {stats['margin']:.4f}")
    print(f"Recall@1:         {stats['recall_at_1']*100:.1f}%")

    # Show a few best/worst examples
    sim = stats["sim_matrix"].cpu()
    top = sim.diag().topk(k=min(3, B)).indices.tolist()
    worst = sim.diag().topk(k=min(3, B), largest=False).indices.tolist()
    def show(idx):
        row = sim[idx]
        pos = row[idx].item()
        hard_neg = row[torch.topk(row, k=2).indices[1 if row.argmax()==idx else 0]].item()
        print(f"- idx {idx:3d}: pos={pos:.4f}  hard_neg={hard_neg:.4f}  src={data['meta'][idx]['src_path']}")
    print("\nTop positives:")
    for i in top: show(i)
    print("\nWorst positives:")
    for i in worst: show(i)

if __name__ == "__main__":
    main()
