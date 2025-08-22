# quick_check.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np, torch, pretty_midi
from aria.model import TransformerEMB, ModelConfig
from aria.config import load_model_config
from ariautils.tokenizer import AbsTokenizer
from safetensors.torch import load_file




# from phrase2context_clip_new import (
#     slice_midi, load_downbeats,
#     embed_pm_segments_fast,           # the fixed version above
#     _aria_forward,               # your forward adapter
#     batched_global_embeddings_from_paths,  # already added earlier
#     _tok_ids_mask_from_pm
# )

ARIA = "aria_ckpts/model.safetensors"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_BARS = 12

# Load Aria
cfg = ModelConfig(**load_model_config(name="medium-emb"))
cfg.set_vocab_size(AbsTokenizer().vocab_size)
aria = TransformerEMB(cfg); aria.load_state_dict(load_file(ARIA), strict=True)
aria.to(DEVICE).eval()

tok = AbsTokenizer()

import io, json, tarfile, glob, os
from pathlib import Path
import numpy as np
import pretty_midi

# --- config ---
SHARDS_DIR = "./shards"   # directory that contains tokens-00000.tar.gz

# --- robust downbeats mirror (same fallback you used) ---
def load_downbeats(pm: pretty_midi.PrettyMIDI):
    try:
        db = pm.get_downbeats()
        if db is not None and len(db) >= 2:
            return np.asarray(db, dtype=np.float32)
    except Exception:
        pass
    # fallback 4/4 @ 120 BPM
    total = pm.get_end_time()
    if total <= 0:  # degenerate
        return np.array([0.0, 1.0], dtype=np.float32)
    bar_sec = 2.0  # 4/4 @ 120bpm → 0.5s/beat * 4
    n_bars = max(2, int(np.ceil(total / bar_sec)) + 1)
    return (np.arange(n_bars, dtype=np.float32) * bar_sec)

# --- find the first shard ---
def first_shard(shards_dir: str|Path) -> Path:
    candidates = sorted(
        [Path(p) for p in glob.glob(os.path.join(str(shards_dir), "tokens-*.tar*"))]
    )
    if not candidates:
        raise FileNotFoundError("No shards found matching tokens-*.tar*")
    return candidates[0]


# --- read one item (meta + sibling npys) from a shard ---
def read_one_item_from_shard(shard_path: Path):
    with tarfile.open(shard_path, "r:*") as tf:
        # pick the first meta.json entry
        metas = [m for m in tf.getmembers() if m.name.endswith(".meta.json")]
        if not metas:
            raise RuntimeError("Shard has no *.meta.json entries")
        m = metas[0]
        stem = m.name[:-len(".meta.json")]  # e.g. 'UID' or 'subdir/UID'

        def get_bytes(name: str):
            # guard in case tar has dirs; tarfile.getmember raises if missing
            names = set(tf.getnames())
            if name not in names:
                raise KeyError(f"Missing entry in tar: {name}")
            fobj = tf.extractfile(name)
            return fobj.read() if fobj is not None else None

        meta = json.load(io.BytesIO(get_bytes(m.name)))
        pid = f"{stem}.phrase_ids.npy"
        pat = f"{stem}.phrase_attn.npy"
        cid = f"{stem}.context_ids.npy"
        cat = f"{stem}.context_attn.npy"

        phr_ids = np.load(io.BytesIO(get_bytes(pid)))
        phr_attn = np.load(io.BytesIO(get_bytes(pat)))
        ctx_ids = np.load(io.BytesIO(get_bytes(cid)))
        ctx_attn = np.load(io.BytesIO(get_bytes(cat)))

        return {
            "uid": Path(stem).name,
            "meta": meta,
            "phrase_ids": phr_ids,
            "phrase_attn": phr_attn,
            "context_ids": ctx_ids,
            "context_attn": ctx_attn,
        }

# --- compute phrase/context time ranges from meta + source MIDI ---
def compute_times_from_meta(meta: dict):
    src_path = meta["src_path"]
    start_bar = int(meta["start_bar"])
    k = int(meta["k_bars"])
    phrase_bars = int(meta["phrase_bars"])

    pm = pretty_midi.PrettyMIDI(src_path)
    downbeats = load_downbeats(pm)

    # context covers bars [start_bar - k, start_bar)
    ctx_start_t = float(downbeats[start_bar - k])
    ctx_end_t   = float(downbeats[start_bar])

    # phrase covers bars [start_bar, start_bar + phrase_bars)
    phr_start_t = float(downbeats[start_bar])
    phr_end_t   = float(downbeats[start_bar + phrase_bars])

    return dict(
        src_path=src_path,
        ctx=(ctx_start_t, ctx_end_t),
        phrase=(phr_start_t, phr_end_t),
        n_downbeats=len(downbeats),
    )
def list_source_midis(shard_path: Path):
    srcs = []
    with tarfile.open(shard_path, "r:*") as tf:
        metas = [m for m in tf.getmembers() if m.name.endswith(".meta.json")]
        for m in metas:
            fobj = tf.extractfile(m)
            if fobj is None:
                continue
            meta = json.load(fobj)
            srcs.append(meta["src_path"])
    return srcs


shard = first_shard(SHARDS_DIR)
item = read_one_item_from_shard(shard)
print(f"item: {item}")
exit()
srcs = list_source_midis(shard)
print(f"Total items in shard: {len(srcs)}")
print(f"Unique source MIDIs: {len(set(srcs))}")


times = compute_times_from_meta(item["meta"])

print("UID:", item["uid"])
print("Source MIDI:", times["src_path"])
print("Phrase ids len / attn len:", len(item["phrase_ids"]), len(item["phrase_attn"]))
print("Context ids len / attn len:", len(item["context_ids"]), len(item["context_attn"]))
print("Phrase time window [s]:", times["phrase"])
print("Context time window [s]:", times["ctx"])


