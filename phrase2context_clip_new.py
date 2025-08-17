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

from safetensors.torch import load_file

from tqdm import tqdm

# --- Aria embedding backbone ---
from aria.embedding import get_global_embedding_from_midi, get_global_embeddings_from_midi_batch, MidiDict
from aria.model import TransformerEMB, ModelConfig
from aria.config import load_model_config
from ariautils.tokenizer import AbsTokenizer
from concurrent.futures import ProcessPoolExecutor, as_completed
import io
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
def _token_budget_batches(token_lens, budget: int):
    """Yield (start, end) ranges of indices so that sum(token_lens[start:end]) <= budget."""
    n = len(token_lens)
    i = 0
    while i < n:
        acc = 0
        j = i
        while j < n and acc + token_lens[j] <= budget:
            acc += token_lens[j]
            j += 1
        if j == i:  # single huge item; ensure forward progress
            j += 1
        yield i, j
        i = j

def _coerce_int_list(lst):
    """
    Ensure a list/array/tuple of tokens or masks is a Python list[int].
    Accepts numpy arrays, lists of str, lists of np types, etc.
    """
    if lst is None:
        return []
    # numpy -> list
    try:
        import numpy as _np
        if isinstance(lst, _np.ndarray):
            lst = lst.tolist()
    except Exception:
        pass
    if not isinstance(lst, (list, tuple)):
        # Some tokenizers return tensors — handle those too
        try:
            lst = list(lst)
        except Exception:
            lst = [lst]
    # Cast each element to int
    out = []
    for x in lst:
        # handle e.g. np.int32, np.int64, torch tensors scalars, strings
        try:
            out.append(int(x))
        except Exception:
            # last resort: strip then int
            out.append(int(str(x).strip()))
    return out


def _tokenize_one(seg_bytes: bytes):
    import pretty_midi, io, tempfile, os as _os
    from ariautils.tokenizer import AbsTokenizer

    tok = AbsTokenizer()

    # bytes -> PrettyMIDI (fallback to temp file if needed)
    try:
        pm = pretty_midi.PrettyMIDI(io.BytesIO(seg_bytes))
    except TypeError:
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            tmp.write(seg_bytes)
            tmp_path = tmp.name
        try:
            pm = pretty_midi.PrettyMIDI(tmp_path)
        finally:
            try: _os.unlink(tmp_path)
            except OSError: pass

    pair = _tok_ids_mask_from_pm(pm, tok)
    if pair is None:
        return None
    ids, mask = pair

    # ---- ensure IDs are integers ----
    def _map_token_to_id(t):
        # already int-like
        try:
            return int(t)
        except Exception:
            pass
        # tokenizer-specific mappings (try a few common patterns)
        if hasattr(tok, "token_to_id"):
            v = tok.token_to_id(t)
            if v is not None:
                return int(v)
        if hasattr(tok, "vocab") and isinstance(tok.vocab, dict):
            if t in tok.vocab:
                return int(tok.vocab[t])
        if hasattr(tok, "stoi") and isinstance(tok.stoi, dict):
            if t in tok.stoi:
                return int(tok.stoi[t])
        if hasattr(tok, "token2id") and isinstance(tok.token2id, dict):
            if t in tok.token2id:
                return int(tok.token2id[t])
        # last resort: unknown token → eos if available, else 0
        try:
            return int(getattr(tok, "eos_tok"))
        except Exception:
            return 0

    # map all ids to ints if any are non-ints
    if ids and not isinstance(ids[0], int):
        ids = [_map_token_to_id(t) for t in ids]
    else:
        ids = [int(x) for x in ids]  # also coerces numpy scalars/strings like "42"

    # coerce mask to ints and align length
    mask = [int(x) for x in (mask or [])]
    if len(mask) != len(ids):
        L = len(ids)
        mask = (mask + [1] * L)[:L]

    # clamp length if your code uses MAX_EMBEDDING_SEQ_LEN
    max_len = globals().get("MAX_EMBEDDING_SEQ_LEN", None)
    if max_len is not None:
        ids  = ids[:max_len]
        mask = mask[:max_len]

    # restore EOS if truncated
    try:
        eos_id = int(getattr(tok, "eos_tok"))
    except Exception:
        eos_id = ids[-1] if ids else 0
    if ids and ids[-1] != eos_id:
        ids[-1] = eos_id

    return (ids, mask)

@torch.no_grad()
def get_pooled_base_embeddings_from_pm_batch(
        pm_list: List[pretty_midi.PrettyMIDI],
        tok: AbsTokenizer,
        aria_model: torch.nn.Module,
        device: str = "cuda",
        max_len_pad: Optional[int] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        use_amp: bool = True,
) -> torch.Tensor:
    """
    Tokenize -> pad -> Aria forward -> masked pool.
    Returns a float32 tensor of shape [B, D] on CPU, in the same order as pm_list.
    Skips empty segments (no pitched notes) by returning a zero-length result for those;
    caller can pre-filter if desired.
    """
    device = torch.device(device)
    aria_model.eval().to(device)

    # 1) Tokenize on CPU
    items = []
    for i, pm in enumerate(pm_list):
        # skip empty segments
        if (not pm.instruments) or all(len(inst.notes) == 0 for inst in pm.instruments):
            items.append((i, None, None))
            continue
        pair = _tok_ids_mask_from_pm(pm, tok)
        if pair is None:
            items.append((i, None, None))
            continue
        ids, mask = pair
        if not ids or not any(mask):
            items.append((i, None, None))
            continue
        items.append((i, torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)))

    # Keep only valid entries; remember original positions
    valid = [(i, ids, msk) for (i, ids, msk) in items if ids is not None]
    if not valid:
        return torch.empty(0, 0, dtype=torch.float32)

    # 2) Bucket by length to reduce padding
    valid.sort(key=lambda t: t[1].numel())
    order = [i for (i, _, _) in valid]

    # 3) Mini collate that pads to the longest item in batch
    def make_batches(seq, bs):
        for s in range(0, len(seq), bs):
            yield seq[s:s+bs]

    pooled_cpu = {}
    amp_ctx = torch.cuda.amp.autocast if (use_amp and device.type == "cuda") else torch.autocast
    with torch.no_grad():
        for chunk in make_batches(valid, batch_size):
            lengths = [x[1].numel() for x in chunk]
            L = max(lengths)
            if max_len_pad is not None:
                L = min(L, max_len_pad)

            B = len(chunk)
            ids_pad  = torch.zeros(B, L, dtype=torch.long, pin_memory=True)
            msk_pad  = torch.zeros(B, L, dtype=torch.long, pin_memory=True)

            for bi, (_, ids, msk) in enumerate(chunk):
                l = min(ids.numel(), L)
                ids_pad[bi, :l] = ids[:l]
                msk_pad[bi, :l] = msk[:l]

            ids_pad  = ids_pad.to(device, non_blocking=True)
            msk_pad  = msk_pad.to(device, non_blocking=True)

            with (torch.cuda.amp.autocast(enabled=(device.type=="cuda" and use_amp))):
                seq = _aria_forward(aria_model, ids_pad, msk_pad)   # [B,L,D] or [B,D]
            if seq.dim() == 3:
                m = msk_pad.unsqueeze(-1).float()
                pooled = (seq * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)  # [B,D]
            else:
                pooled = seq  # [B,D]

            pooled = pooled.detach().float().cpu()
            for (bi, (orig_idx, _, _)) in enumerate(chunk):
                pooled_cpu[orig_idx] = pooled[bi]

    # 4) Reassemble in original pm_list order (skipping empties)
    outs = []
    for i, ids, m in items:
        if ids is None:
            continue
        outs.append(pooled_cpu[i])
    return torch.stack(outs, dim=0)  # [B', D]

@torch.no_grad()
def get_pooled_base_embeddings_from_token_batch(
        ids_list: List[List[int]],
        mask_list: List[List[int]],
        aria_model: torch.nn.Module,
        device: str = "cuda",
        batch_size: int = 512,
        use_amp: bool = True,
) -> torch.Tensor:
    device_t = torch.device(device)
    aria_model.eval().to(device_t)

    outs = []

    def chunks(n):
        for i in range(0, len(ids_list), n):
            yield i, min(i+n, len(ids_list))

    for i, j in chunks(batch_size):
        chunk_ids  = ids_list[i:j]
        chunk_mask = mask_list[i:j]
        L = max(len(s) for s in chunk_ids)
        ids = torch.zeros(j - i, L, dtype=torch.long, pin_memory=True)
        msk = torch.zeros(j - i, L, dtype=torch.long, pin_memory=True)
        for bi, (s, m) in enumerate(zip(chunk_ids, chunk_mask)):
            # force both to list[int]
            s = [int(x) for x in s]
            m = [int(x) for x in m]
            if len(m) != len(s):
                m = (m + [1] * len(s))[:len(s)]
            if len(s) == 0:
                continue
            ids[bi, :len(s)] = torch.tensor(s, dtype=torch.long)
            msk[bi, :len(s)] = torch.tensor(m, dtype=torch.long)


        ids = ids.to(device_t, non_blocking=True)
        msk = msk.to(device_t, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(use_amp and device_t.type == "cuda")):
            out = _aria_forward(aria_model, ids, msk)
        if isinstance(out, (tuple, list)):
            out = out[0]

        m = msk.unsqueeze(-1).float()
        pooled = (out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)
        outs.append(pooled)

    return torch.cat(outs, dim=0)

def _tokenize_pm_list(pm_list: List[pretty_midi.PrettyMIDI], tok: AbsTokenizer):
    """Return ids_list, mask_list, kept_idx (only valid/nonnull)."""
    ids_list, mask_list, kept = [], [], []
    for idx, pm in enumerate(pm_list):
        if (not pm.instruments) or all(len(inst.notes) == 0 for inst in pm.instruments):
            continue
        pair = _tok_ids_mask_from_pm(pm, tok)
        if pair is None:
            continue
        ids, mask = pair
        # coerce + clamp + EOS fix (same as encode-index)
        ids  = [int(x) for x in ids][:globals().get("MAX_EMBEDDING_SEQ_LEN", len(ids))]
        mask = [int(x) for x in (mask or [])][:len(ids)]
        if len(mask) != len(ids):
            mask = (mask + [1]*len(ids))[:len(ids)]
        try:
            eos = int(getattr(tok, "eos_tok"))
        except Exception:
            eos = ids[-1] if ids else 0
        if ids and ids[-1] != eos:
            ids[-1] = eos
        if not ids or not any(mask):
            continue
        ids_list.append(ids)
        mask_list.append(mask)
        kept.append(idx)
    return ids_list, mask_list, kept

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

    def forward(self, batch_segments: List[Tuple[pretty_midi.PrettyMIDI, pretty_midi.PrettyMIDI]]):
        tok = AbsTokenizer()

        pm_phr = [p for (p, _) in batch_segments]
        pm_ctx = [c for (_, c) in batch_segments]

        # Tokenize both sides
        ids_p, m_p, kept_p = _tokenize_pm_list(pm_phr, tok)
        ids_c, m_c, kept_c = _tokenize_pm_list(pm_ctx, tok)

        # Align by original indices: keep only indices present in BOTH
        set_p, set_c = set(kept_p), set(kept_c)
        common = sorted(set_p & set_c)
        if len(common) < 2:
            # not enough aligned pairs for a stable InfoNCE step
            return (torch.empty(0, 256, device=self.device),
                    torch.empty(0, 256, device=self.device))

        # Build aligned token lists in the same order
        idxpos_p = {k:i for i,k in enumerate(kept_p)}
        idxpos_c = {k:i for i,k in enumerate(kept_c)}
        ids_p_aligned  = [ids_p[idxpos_p[k]] for k in common]
        m_p_aligned    = [m_p[idxpos_p[k]]   for k in common]
        ids_c_aligned  = [ids_c[idxpos_c[k]] for k in common]
        m_c_aligned    = [m_c[idxpos_c[k]]   for k in common]

        # Batched forward on GPU
        base_phrase = get_pooled_base_embeddings_from_token_batch(
            ids_list=ids_p_aligned, mask_list=m_p_aligned,
            aria_model=self.aria, device=str(self.device),
            batch_size=1024, use_amp=True,
        ).to(self.device, non_blocking=True)

        base_context = get_pooled_base_embeddings_from_token_batch(
            ids_list=ids_c_aligned, mask_list=m_c_aligned,
            aria_model=self.aria, device=str(self.device),
            batch_size=1024, use_amp=True,
        ).to(self.device, non_blocking=True)

        zp = self.phrase_head(base_phrase)
        zc = self.context_head(base_context)
        zp = nn.functional.normalize(zp, p=2, dim=1, eps=1e-6)
        zc = nn.functional.normalize(zc, p=2, dim=1, eps=1e-6)
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
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    global_step = 0
    try:
        for epoch in range(args.epochs):
            pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}", ncols=100)
            for batch_segments in pbar:
                if len(batch_segments) < args.batch_size:
                    continue
                zp, zc = model(batch_segments)
                with torch.no_grad():
                    zz = torch.cat([zp, zc], dim=0).detach().cpu().numpy()
                    print("[train dbg] proj_feature_std_mean:", zz.std(axis=0).mean())
                loss = clip_loss(zp, zc, temperature=args.temp)

                opt.zero_grad()
                loss.backward()
                print([p.grad.norm() for p in model.phrase_head.parameters()])
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
    except KeyboardInterrupt:
        print("\nInterrupted — saving last checkpoint to proj_heads_last.pt ...")
        _save_heads(model, out, "last", args.k_bars)
        raise
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

    def _build_head_from_state_dict(head_sd: dict):
        w1 = head_sd['0.weight']  # (hid, base_dim)
        w2 = head_sd['2.weight']  # (proj_dim, hid)
        base_dim = w1.shape[1]
        hid      = w1.shape[0]
        proj_dim = w2.shape[0]
        head = nn.Sequential(
            nn.Linear(base_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, proj_dim)
        ).to(args.device)
        head.load_state_dict(head_sd)
        head.eval()
        return head, base_dim, hid, proj_dim

    if 'context_head' in ckpt:
        context_head, base_dim, hid, proj_dim = _build_head_from_state_dict(ckpt['context_head'])
    else:
        # adjust key if your ckpt uses a different one
        context_head, base_dim, hid, proj_dim = _build_head_from_state_dict(ckpt['head'])

    print(f"[encode] head dims: base={base_dim}, hid={hid}, proj={proj_dim}")
    wstd = next(context_head.parameters()).detach().float().cpu().numpy().std()
    print("[encode dbg] loaded context_head weight std:", wstd)
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
    tok = AbsTokenizer()
    with open(meta_path, "w", encoding="utf-8") as wf:
        for mf in tqdm(midi_files, desc="Encoding contexts"):

            pm = pretty_midi.PrettyMIDI(str(mf))
            db = load_downbeats(pm)
            # --- inside encode_contexts(), per-file block ---
            segs, spans, ids_list, mask_list = [], [], [], []
            for i in range(args.k_bars, len(db) - 4):
                cs, ce = db[i - args.k_bars], db[i]
                seg = slice_midi(pm, cs, ce)
                if (not seg.instruments) or all(len(inst.notes) == 0 for inst in seg.instruments):
                    continue

                pair = _tok_ids_mask_from_pm(seg, tok)  # <-- direct, no BytesIO, no pool
                if pair is None:
                    continue
                ids, mask = pair

                # coerce + clamp + EOS fix
                ids  = [int(x) for x in ids][:globals().get("MAX_EMBEDDING_SEQ_LEN", len(ids))]
                mask = [int(x) for x in (mask or [])][:len(ids)]
                if len(mask) != len(ids):
                    mask = (mask + [1]*len(ids))[:len(ids)]
                try:
                    eos = int(getattr(tok, "eos_tok"))
                except Exception:
                    eos = ids[-1] if ids else 0
                if ids and ids[-1] != eos:
                    ids[-1] = eos

                if not ids or not any(mask):
                    continue

                ids_list.append(ids)
                mask_list.append(mask)
                spans.append((i - args.k_bars, i))

            if not ids_list:
                continue

            # batch tokens -> pooled embeddings on GPU
            base = get_pooled_base_embeddings_from_token_batch(
                ids_list=ids_list,
                mask_list=mask_list,
                aria_model=aria,
                device=args.device,
                batch_size=1024,  # try 1024–2048 if VRAM allows
                use_amp=True,
            )
            base = base.to(args.device, non_blocking=True)
            z = context_head(base)
            z = torch.nn.functional.normalize(z, p=2, dim=1).cpu().numpy().astype("float32")

            z_t = torch.nn.functional.normalize(context_head(base), p=2, dim=1, eps=1e-6)

            z_np = z_t.detach().cpu().numpy()
            print("[encode dbg] base_norm min/mean/max:",
                  base.norm(dim=1).min().item(),
                  base.norm(dim=1).mean().item(),
                  base.norm(dim=1).max().item())
            print("[encode dbg] proj_feature_std_mean:", z_np.std(axis=0).mean())
            print("[encode dbg] unique_rows@1e-5:", len({tuple(np.round(v, 5)) for v in z_np}), "/", len(z_np))

            for (bpair, vec) in zip(spans, z):
                all_vecs.append(vec)
                wf.write(f"{mf}\t{bpair[0]}\t{bpair[1]}\n")

    if not all_vecs:
        print("No context vectors produced. Try lowering --k-bars or enable ENC_DEBUG=1 for tracebacks.")
        return
    arr = np.vstack(all_vecs).astype("float32")
    np.save(vec_path, arr)
    print(f"Wrote {arr.shape} vectors to {vec_path} and metadata to {meta_path}")

# -----------------------------
# Query (phrase -> top-K contexts)
# -----------------------------
@torch.no_grad()
def query(args):
    import faiss

    tok = AbsTokenizer()

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

    # NEW: embed phrase with the same pipeline used in encode-index
    base = get_pooled_base_embeddings_from_pm_batch(
        pm_list=[pm_phr],
        tok=AbsTokenizer(),
        aria_model=aria,
        device=args.device,
        batch_size=1,
        use_amp=True,
    ).to(args.device, non_blocking=True)

    z = phrase_head(base)
    z = torch.nn.functional.normalize(z, p=2, dim=1).cpu().numpy().astype("float32")

    # z is the (1, D) normalized query vector as numpy float32
    # Instead of "top-k rows regardless of file", pick top-k FILES by their best row

    # Ensure xb is L2-normalized (done above with faiss.normalize_L2(xb)).
    # Normalize the query just in case:
    z = z / np.linalg.norm(z, ord=2, axis=1, keepdims=True).clip(1e-12)
    # Cosine similarity to all context vectors
    sims = xb @ z[0]  # (N,)

    # For each file, keep the single best location (max similarity)
    best_by_file = {}  # mf -> dict(similarity, idx)
    for idx, (mf, cst, cen) in enumerate(meta):
        s = float(sims[idx])
        if (mf not in best_by_file) or (s > best_by_file[mf]["similarity"]):
            # store best index + similarity for this file
            best_by_file[mf] = {"similarity": s, "idx": idx}

    # Rank files by their max similarity
    items = sorted(
        best_by_file.items(),
        key=lambda kv: kv[1]["similarity"],
        reverse=True
    )
    items = items[:args.topk]

    # Build results with the best location per file
    results = []
    for rank, (mf, info) in enumerate(items, 1):
        idx = info["idx"]
        _, cst, cen = meta[idx]
        results.append({
            "rank": rank,
            "similarity": float(info["similarity"]),
            "midi_path": mf,
            "best_context_bars": [int(cst), int(cen)],  # location of the max hit in this file
        })

    import json
    print(json.dumps(results, indent=2))

@torch.no_grad()
def embed_pm_segment(
        pm_seg: pretty_midi.PrettyMIDI,
        tok,                      # AbsTokenizer instance (e.g., AbsTokenizer())
        aria_model,               # Aria embedding backbone (eval mode, on device)
        proj_head: torch.nn.Module,  # context_head or phrase_head (eval mode, on device)
        device: str = "cuda",
        skip_if_empty: bool = True,
) -> Optional[np.ndarray]:
    """
    Tokenize a pretty_midi.PrettyMIDI segment, run through Aria backbone, pool, then
    project with the provided head and L2-normalize. Returns a (proj_dim,) float32 numpy
    vector, or None if the segment is empty and skip_if_empty=True.

    Parameters
    ----------
    pm_seg : pretty_midi.PrettyMIDI
        Pre-sliced segment (e.g., a K-bar context or 4-bar phrase).
    tok : AbsTokenizer
        Tokenizer with .encode_pretty_midi(pm) -> (ids, mask).
    aria_model : torch.nn.Module
        Aria embedding model. Must accept input_ids, attention_mask and return (B, L, D) or similar.
    proj_head : torch.nn.Module
        Projection head (e.g., context_head or phrase_head). Outputs (B, proj_dim).
    device : str
        'cuda' or 'cpu'.
    skip_if_empty : bool
        If True, returns None when the segment has no pitched notes.

    Returns
    -------
    Optional[np.ndarray]
        L2-normalized vector of shape (proj_dim,), dtype float32; or None if skipped.
    """
    # Optional: skip segments with no notes (after drum filter)
    if skip_if_empty:
        if (not pm_seg.instruments) or all(len(inst.notes) == 0 for inst in pm_seg.instruments):
            return None

    # Tokenize
    pair = _tok_ids_mask_from_pm(pm_seg, tok)
    if pair is None:
        return None if skip_if_empty else None
    ids, mask = pair
    if not ids or not any(mask):
        return None if skip_if_empty else None

    # Tensors on device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)          # (1, L)
    attention_mask = torch.tensor([mask], dtype=torch.long, device=device)    # (1, L)

    seq = _aria_forward(aria_model, input_ids, attention_mask)  # (1, L, D) or (1, D)

    # Mean-pool with mask → (1, D)
    m = attention_mask.unsqueeze(-1).float()
    pooled = (seq * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)

    # Project + normalize → (1, proj_dim)
    z = proj_head(pooled)
    z = torch.nn.functional.normalize(z, p=2, dim=1)

    # → (proj_dim,) float32 numpy
    return z.squeeze(0).detach().cpu().numpy().astype("float32")

def _tok_ids_mask_from_pm(pm_seg, tok) -> Optional[Tuple[List[int], List[int]]]:
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=True) as tmp:
        pm_seg.write(tmp.name)
        md = MidiDict.from_midi(mid_path=tmp.name)

    if len(md.note_msgs) == 0:
        return None

    tokens = tok.tokenize(md, add_dim_tok=False)
    if tokens is None or len(tokens) == 0:
        return None

    # ensure we have int ids
    ids = tok.encode(tokens)
    if ids is None or len(ids) == 0:
        return None

    mask = [1] * len(ids)
    return ids, mask

def _aria_forward(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """
    Call the Aria embedding backbone regardless of signature differences across revisions.
    Tries several common call signatures and returns a tensor of shape (B, L, D) or (B, D).
    """
    # Most likely: positional args (input_ids, attention_mask)
    try:
        out = model(input_ids, attention_mask)
        return out[0] if isinstance(out, (tuple, list)) else out
    except TypeError:
        pass

    # Some versions: (input_ids,) only
    try:
        out = model(input_ids)
        return out[0] if isinstance(out, (tuple, list)) else out
    except TypeError:
        pass

    # Some versions: keyword names x/mask
    try:
        out = model(x=input_ids, mask=attention_mask)
        return out[0] if isinstance(out, (tuple, list)) else out
    except TypeError:
        pass

    # Fallback: direct forward
    try:
        out = model.forward(input_ids, attention_mask)
        return out[0] if isinstance(out, (tuple, list)) else out
    except TypeError as e:
        raise TypeError(
            "Could not call Aria backbone; unknown forward signature. "
            "Tried (ids,mask), (ids), and (x=,mask=)."
        ) from e

def _save_heads(model, out_dir: Path, tag: str, k_bars: int):
    """Save projection heads with dimensions; backbone is frozen and not saved."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "phrase_head": model.phrase_head.state_dict(),
        "context_head": model.context_head.state_dict(),
        "base_dim": next(model.phrase_head.parameters()).shape[1],
        "proj_dim": model.phrase_head[-1].out_features,
        "k_bars": k_bars
    }, out_dir / f"proj_heads_{tag}.pt")

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
