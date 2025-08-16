'''
NB: The following must be added to aria/__init__.py:

__all__ = [
    'config',
    'datasets',
    'embedding',
    'eval',
    'inference',
    'model',
    'run',
    'training',
    'utils',
]

'''

#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, tempfile, math, random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pretty_midi
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- Aria embedding backbone ---
from aria.embedding import get_global_embedding_from_midi  # public helper
from aria.model import TransformerEMB, ModelConfig
from aria.config import load_model_config
from ariautils.tokenizer import AbsTokenizer
from safetensors.torch import load_file

# -----------------------------
# Utilities: MIDI slicing & transposition
# -----------------------------
def load_downbeats(pm: pretty_midi.PrettyMIDI) -> np.ndarray:
    # Robust across tempo / TS changes; returns np.ndarray of downbeat times (seconds).
    db = pm.get_downbeats()
    if db is None or len(db) < 2:
        # Fallback: assume 4/4 at first tempo
        change_times, tempi = pm.get_tempo_changes()
        bpm = float(tempi[0]) if len(tempi) else 120.0
        bar_sec = 4.0 * 60.0 / bpm
        end = pm.get_end_time()
        n_bars = max(2, int(math.ceil(end / bar_sec)) + 1)
        return np.linspace(0.0, n_bars * bar_sec, n_bars + 1)
    return db

def slice_midi(pm: pretty_midi.PrettyMIDI, start: float, end: float) -> pretty_midi.PrettyMIDI:
    """Return a new PrettyMIDI object containing notes overlapping [start, end), times shifted to start at 0."""
    out = pretty_midi.PrettyMIDI()
    for inst in pm.instruments:
        if inst.is_drum:  # we ignore drums for Aria's piano-centric embeddings
            continue
        new_inst = pretty_midi.Instrument(program=inst.program, is_drum=False, name=inst.name)
        for n in inst.notes:
            if n.end <= start or n.start >= end:
                continue
            s = max(n.start, start) - start
            e = min(n.end, end) - start
            if e > s:
                new_inst.notes.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=s, end=e))
        if new_inst.notes:
            out.instruments.append(new_inst)
    # Keep it simple: we don't copy tempo/TS changes; embeddings are robust to that
    return out

def transpose_inplace(pm: pretty_midi.PrettyMIDI, semitones: int):
    if semitones == 0: return
    for inst in pm.instruments:
        if inst.is_drum: continue
        for n in inst.notes:
            n.pitch = int(np.clip(n.pitch + semitones, 0, 127))

# -----------------------------
# Dataset
# -----------------------------
class PhraseContextPairs(Dataset):
    """
    Yields pairs: (midi_path, (phrase_start_t, phrase_end_t), (ctx_start_t, ctx_end_t))
    where phrase is 4 bars and context is the K bars immediately preceding.
    """
    def __init__(self,
                 midi_files: List[Path],
                 k_bars: int = 8,
                 phrase_bars: int = 4,
                 stride_bars: int = 1,
                 max_pairs_per_file: Optional[int] = None,
                 seed: int = 17,
                 augment_transpose_steps: int = 0):
        self.k_bars = k_bars
        self.phrase_bars = phrase_bars
        self.stride_bars = stride_bars
        self.max_pairs_per_file = max_pairs_per_file
        self.augment_transpose_steps = augment_transpose_steps
        self.rng = random.Random(seed)

        self.items: List[Tuple[Path, float, float, float, float, int]] = []
        for mf in midi_files:
            try:
                pm = pretty_midi.PrettyMIDI(str(mf))
                db = load_downbeats(pm)
                # windows like: [i-k, i) -> context, [i, i+4) -> phrase
                # i runs where both segments fit inside db
                candidates: List[Tuple[float,float,float,float]] = []
                for i in range(self.k_bars, len(db) - self.phrase_bars):
                    ctx_st, ctx_en = db[i - self.k_bars], db[i]
                    phr_st, phr_en = db[i], db[i + self.phrase_bars]
                    candidates.append((phr_st, phr_en, ctx_st, ctx_en))
                # stride and cap
                candidates = candidates[::max(1, self.stride_bars)]
                if self.max_pairs_per_file is not None and len(candidates) > self.max_pairs_per_file:
                    candidates = self.rng.sample(candidates, self.max_pairs_per_file)
                # (path, phrase_start, phrase_end, ctx_start, ctx_end, transpose_steps)
                for (ps, pe, cs, ce) in candidates:
                    t = 0
                    if self.augment_transpose_steps > 0:
                        t = self.rng.randint(-self.augment_transpose_steps, self.augment_transpose_steps)
                    self.items.append((mf, ps, pe, cs, ce, t))
            except Exception:
                continue

    def __len__(self): return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]

# -----------------------------
# Model: Aria backbone + projections
# -----------------------------
class DualEncoder(nn.Module):
    """
    Uses a frozen Aria TransformerEMB to produce base embeddings.
    Two small projection heads (phrase/context) map base_emb -> 256-dim, L2-normalized.
    """
    def __init__(self, aria_ckpt_path: str, device: str = "cuda"):
        super().__init__()
        # Load Aria embedding model per README (kept consistent with repo)
        model_cfg = ModelConfig(**load_model_config(name="medium-emb"))
        model_cfg.set_vocab_size(AbsTokenizer().vocab_size)
        self.aria = TransformerEMB(model_cfg)
        sd = load_file(filename=aria_ckpt_path)
        self.aria.load_state_dict(sd, strict=True)
        self.device = torch.device(device)
        self.aria.to(self.device)
        self.aria.eval()  # backbone frozen in Stage-A

        # Determine base embedding size by making a tiny dummy call after init
        # We won't actually forward tokens here; instead, infer from config if available
        base_dim = 512

        hid = 512
        out = 256
        self.phrase_head = nn.Sequential(nn.Linear(base_dim, hid), nn.ReLU(), nn.Linear(hid, out))
        self.context_head = nn.Sequential(nn.Linear(base_dim, hid), nn.ReLU(), nn.Linear(hid, out))

    @torch.no_grad()
    def _embed_segment(self, pm: pretty_midi.PrettyMIDI) -> torch.Tensor:
        # write to a temp file and use aria.embeddings helper
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=True) as tmp:
            pm.write(tmp.name)
            emb = get_global_embedding_from_midi(
                model=self.aria,
                midi_path=tmp.name,
                device=str(self.device) if self.device.type == "cuda" else "cpu"
            )
        # ensure torch tensor on device
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb)
        return emb.to(self.device).float()

    def forward(self, batch_segments: List[Tuple[pretty_midi.PrettyMIDI, pretty_midi.PrettyMIDI]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs: list of (pm_phrase, pm_context) objects (already sliced & transposed)
        Returns: (Zp, Zc) in R^{B x 256}, L2-normalized
        """
        base_phrase = []
        base_context = []
        # (No grad for backbone in Stage-A)
        with torch.no_grad():
            for pm_p, pm_c in batch_segments:
                base_phrase.append(self._embed_segment(pm_p))
                base_context.append(self._embed_segment(pm_c))
        base_phrase = torch.stack(base_phrase, dim=0)  # (B, D)
        base_context = torch.stack(base_context, dim=0)  # (B, D)

        zp = self.phrase_head(base_phrase)
        zc = self.context_head(base_context)
        # L2 normalize
        zp = nn.functional.normalize(zp, p=2, dim=1)
        zc = nn.functional.normalize(zc, p=2, dim=1)
        return zp, zc

# -----------------------------
# Loss: symmetric InfoNCE (CLIP)
# -----------------------------
def clip_loss(zp: torch.Tensor, zc: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    # zp, zc: (B, D), already L2-normalized
    logits = (zp @ zc.t()) / temperature      # (B,B)
    labels = torch.arange(zp.size(0), device=zp.device)
    loss_p = nn.functional.cross_entropy(logits, labels)
    loss_c = nn.functional.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_p + loss_c)

# -----------------------------
# Collate: slice & augment on the fly
# -----------------------------
def make_collate(k_bars: int, phrase_bars: int, transpose_range: int = 0):
    def _collate(samples: List[Tuple[Path, float, float, float, float, int]]):
        batch_segments = []
        for (mf, ps, pe, cs, ce, tsteps) in samples:
            try:
                pm = pretty_midi.PrettyMIDI(str(mf))
                pm_phrase = slice_midi(pm, ps, pe)
                pm_context = slice_midi(pm, cs, ce)
                # transposition augmentation: apply same shift to both
                if transpose_range > 0 and tsteps != 0:
                    transpose_inplace(pm_phrase, tsteps)
                    transpose_inplace(pm_context, tsteps)
                batch_segments.append((pm_phrase, pm_context))
            except Exception:
                continue
        return batch_segments
    return _collate

# -----------------------------
# Training driver
# -----------------------------
def train(args):
    midi_root = Path(args.midi_root)
    midi_files = (list(midi_root.rglob("*.mid")) +
                  list(midi_root.rglob("*.midi")) +
                  list(midi_root.rglob("*.MID")) +
                  list(midi_root.rglob("*.MIDI")))
    if args.max_files is not None:
        midi_files = midi_files[:args.max_files]

    ds = PhraseContextPairs(
        midi_files=midi_files,
        k_bars=args.k_bars,
        phrase_bars=4,
        stride_bars=args.stride_bars,
        max_pairs_per_file=args.max_pairs_per_file,
        augment_transpose_steps=args.transpose
    )

    print(f"Training pairs: {len(ds):,}")

    model = DualEncoder(aria_ckpt_path=args.aria_ckpt, device=args.device)
    model.to(args.device)

    opt = torch.optim.AdamW(list(model.phrase_head.parameters()) + list(model.context_head.parameters()),
                            lr=args.lr, weight_decay=1e-4)

    collate = make_collate(k_bars=args.k_bars, phrase_bars=4, transpose_range=args.transpose)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, collate_fn=collate, drop_last=True, pin_memory=True)

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}", ncols=100)
        for batch_segments in pbar:
            if len(batch_segments) < args.batch_size:
                continue
            zp, zc = model(batch_segments)
            loss = clip_loss(zp, zc, temperature=args.temp)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.phrase_head.parameters()) + list(model.context_head.parameters()), max_norm=1.0)
            opt.step()

            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # save projection heads (backbone is frozen)
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        torch.save({
            "phrase_head": model.phrase_head.state_dict(),
            "context_head": model.context_head.state_dict(),
            "base_dim": next(model.phrase_head.parameters()).shape[1],
            "proj_dim": model.phrase_head[-1].out_features,
            "k_bars": args.k_bars
        }, out / f"proj_heads_epoch{epoch+1}.pt")
    print("Done.")

# -----------------------------
# Encoding contexts for indexing
# -----------------------------
@torch.no_grad()
def encode_contexts(args):
    # Load model + heads
    model_cfg = ModelConfig(**load_model_config(name="medium-emb"))
    model_cfg.set_vocab_size(AbsTokenizer().vocab_size)
    aria = TransformerEMB(model_cfg)
    sd = load_file(filename=args.aria_ckpt)
    aria.load_state_dict(sd, strict=True)
    aria.to(args.device).eval()

    # projection heads
    ckpt = torch.load(args.proj_ckpt, map_location=args.device)
    base_dim = ckpt["base_dim"]; proj_dim = ckpt["proj_dim"]
    hid = 512
    context_head = nn.Sequential(nn.Linear(base_dim, hid), nn.ReLU(), nn.Linear(hid, proj_dim)).to(args.device)
    context_head.load_state_dict(ckpt["context_head"]); context_head.eval()

    midi_root = Path(args.midi_root)
    midi_files = (list(midi_root.rglob("*.mid")) +
                  list(midi_root.rglob("*.midi")) +
                  list(midi_root.rglob("*.MID")) +
                  list(midi_root.rglob("*.MIDI")))
    if args.max_files is not None:
        midi_files = midi_files[:args.max_files]

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "contexts.tsv"
    vec_path = out_dir / "contexts.f32.npy"

    all_vecs = []
    with open(meta_path, "w", encoding="utf-8") as wf:
        for mf in tqdm(midi_files, desc="Encoding contexts"):
            try:
                pm = pretty_midi.PrettyMIDI(str(mf))
                db = load_downbeats(pm)
                for i in range(args.k_bars, len(db) - 4):
                    cs, ce = db[i - args.k_bars], db[i]
                    pm_ctx = slice_midi(pm, cs, ce)
                    with tempfile.NamedTemporaryFile(suffix=".mid", delete=True) as tmp:
                        pm_ctx.write(tmp.name)
                        base = get_global_embedding_from_midi(aria, tmp.name, device=args.device)
                    if not isinstance(base, torch.Tensor):
                        base = torch.tensor(base)
                    z = context_head(base.to(args.device).float()[None, ...])  # (1,D)
                    z = nn.functional.normalize(z, p=2, dim=1)
                    all_vecs.append(z.squeeze(0).cpu().numpy().astype("float32"))
                    wf.write(f"{mf}\t{i-args.k_bars}\t{i}\n")
            except Exception:
                continue

    arr = np.vstack(all_vecs).astype("float32")
    np.save(vec_path, arr)
    print(f"Wrote {arr.shape} vectors to {vec_path} and metadata to {meta_path}")

# -----------------------------
# Query (phrase -> top-K contexts)
# -----------------------------
@torch.no_grad()
def query(args):
    import faiss

    # load indexable vectors
    vecs = np.load(Path(args.index_dir) / "contexts.f32.npy")
    meta = [line.strip().split("\t") for line in open(Path(args.index_dir) / "contexts.tsv", "r", encoding="utf-8")]

    # build or load FAISS (exact IP on L2-normalized vectors == cosine)
    xb = vecs  # (N,D)
    d = xb.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(xb)
    index.add(xb)

    # load backbone + phrase head
    model_cfg = ModelConfig(**load_model_config(name="medium-emb"))
    model_cfg.set_vocab_size(AbsTokenizer().vocab_size)
    aria = TransformerEMB(model_cfg)
    sd = load_file(filename=args.aria_ckpt)
    aria.load_state_dict(sd, strict=True)
    aria.to(args.device).eval()

    ckpt = torch.load(args.proj_ckpt, map_location=args.device)
    base_dim = ckpt["base_dim"]; proj_dim = ckpt["proj_dim"]
    hid = 512
    phrase_head = nn.Sequential(nn.Linear(base_dim, hid), nn.ReLU(), nn.Linear(hid, proj_dim)).to(args.device)
    phrase_head.load_state_dict(ckpt["phrase_head"]); phrase_head.eval()

    # embed the query phrase (first 4 bars of the prompt MIDI, or use --bars to select)
    pm = pretty_midi.PrettyMIDI(str(args.prompt))
    db = load_downbeats(pm)
    i0 = 0 if args.start_bar is None else max(0, args.start_bar)
    ps, pe = db[i0], db[i0 + 4]
    pm_phr = slice_midi(pm, ps, pe)

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=True) as tmp:
        pm_phr.write(tmp.name)
        base = get_global_embedding_from_midi(aria, tmp.name, device=args.device)
    if not isinstance(base, torch.Tensor):
        base = torch.tensor(base)
    z = phrase_head(base.to(args.device).float()[None, ...])
    z = nn.functional.normalize(z, p=2, dim=1).cpu().numpy().astype("float32")

    D, I = index.search(z, args.topk)
    results = []
    for rank, (d, idx) in enumerate(zip(D[0], I[0]), 1):
        mf, cst, cen = meta[idx]
        results.append({"rank": rank, "similarity": float(d), "midi_path": mf,
                        "context_bars": [int(cst), int(cen)]})
    import json
    print(json.dumps(results, indent=2))

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Directional CLIP-style phrase->context with Aria")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--midi-root", type=Path, required=True)
    t.add_argument("--aria-ckpt", type=str, required=True, help="path to aria-medium-embedding safetensors")
    t.add_argument("--out-dir", type=Path, default=Path("runs/phrase2ctx"))
    t.add_argument("--k-bars", type=int, default=8)
    t.add_argument("--stride-bars", type=int, default=1)
    t.add_argument("--max-files", type=int, default=None)
    t.add_argument("--max-pairs-per-file", type=int, default=64)
    t.add_argument("--transpose", type=int, default=2, help="±semitones")
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--temp", type=float, default=0.05)
    t.add_argument("--num-workers", type=int, default=4)
    t.add_argument("--device", type=str, default="cuda")

    e = sub.add_parser("encode-index")
    e.add_argument("--midi-root", type=Path, required=True)
    e.add_argument("--aria-ckpt", type=str, required=True)
    e.add_argument("--proj-ckpt", type=Path, required=True)
    e.add_argument("--out-dir", type=Path, default=Path("indices/ctx"))
    e.add_argument("--k-bars", type=int, default=8)
    e.add_argument("--max-files", type=int, default=None)
    e.add_argument("--device", type=str, default="cuda")

    q = sub.add_parser("query")
    q.add_argument("--index-dir", type=Path, required=True)
    q.add_argument("--aria-ckpt", type=str, required=True)
    q.add_argument("--proj-ckpt", type=Path, required=True)
    q.add_argument("--prompt", type=Path, required=True)
    q.add_argument("--topk", type=int, default=10)
    q.add_argument("--start-bar", type=int, default=None)
    q.add_argument("--device", type=str, default="cuda")

    args = ap.parse_args()
    if args.cmd == "train": train(args)
    elif args.cmd == "encode-index": encode_contexts(args)
    else: query(args)

if __name__ == "__main__":
    main()
