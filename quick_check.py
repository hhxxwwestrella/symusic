# quick_check.py
import numpy as np, torch, pretty_midi
from pathlib import Path
from aria.model import TransformerEMB, ModelConfig
from aria.config import load_model_config
from ariautils.tokenizer import AbsTokenizer
from safetensors.torch import load_file
import torch.nn as nn
from aria.embedding import MidiDict

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

# # Collect a few context segments
# files = sorted(Path("mini_midis").glob("*.mid"))[:5]
# pm_segments = []
# names = []
# for mf in files:
#     pm = pretty_midi.PrettyMIDI(str(mf))
#     db = load_downbeats(pm)
#     if len(db) >= K_BARS + 1:
#         pm_ctx = slice_midi(pm, db[0], db[K_BARS])
#         pm_segments.append(pm_ctx); names.append(mf.name)
#
# # Run batched embedding (global emb per segment)
# vecs = embed_pm_segments(aria, tok, pm_segments, device=DEVICE)   # (N, D)
# vecs_np = vecs.detach().cpu().numpy()
#
# print("files tested:", names)
# print("shape:", vecs_np.shape)
# print("pairwise L2 distance from vec[0]:", [float(np.linalg.norm(vecs_np[0]-v)) for v in vecs_np])
# print("variance across all vectors:", float(vecs_np.var()))
#
# from aria.embedding import get_global_embedding_from_midi
#
# # compare first sample
# import tempfile, os
# ramdir = "/dev/shm" if os.path.isdir("/dev/shm") else None
# with tempfile.NamedTemporaryFile(suffix=".mid", dir=ramdir, delete=True) as tmp:
#     pm_segments[0].write(tmp.name)
#     base_helper = get_global_embedding_from_midi(model=aria, midi_path=tmp.name, device=DEVICE)
#     if not isinstance(base_helper, torch.Tensor):
#         base_helper = torch.tensor(base_helper, device=DEVICE)
#     base_batched = vecs[0].to(DEVICE)
#     cos = torch.nn.functional.cosine_similarity(base_helper[None], base_batched[None]).item()
#     print("cosine(helper vs batched):", cos)
from phrase2context_clip import DualEncoder
pm = pretty_midi.PrettyMIDI("mini_midis/121437_1.mid")


from ariautils.tokenizer import AbsTokenizer
from phrase2context_clip_new import slice_midi, _tok_ids_mask_from_pm, PhraseContextPairs
import pretty_midi
midi_root = Path("mini_midis")
midi_files = list(midi_root.rglob("*.mid"))
ds = PhraseContextPairs(midi_files=midi_files, k_bars=8, phrase_bars=4)


tok = AbsTokenizer()

# mf, ps, pe, cs, ce, t = ds[0]
# pm = pretty_midi.PrettyMIDI(str(mf))
#
# pm_phrase = slice_midi(pm, ps, pe)
# pm_context = slice_midi(pm, cs, ce)
#
# ids_p, mask_p = _tok_ids_mask_from_pm(pm_phrase, tok)
# ids_c, mask_c = _tok_ids_mask_from_pm(pm_context, tok)
#
# print("Phrase tokens:", ids_p[:50], "... len=", len(ids_p))
# print("Context tokens:", ids_c[:50], "... len=", len(ids_c))

from ariautils.tokenizer import AbsTokenizer
from aria.embedding import _get_chunks, _validate_midi_for_emb


mf = "mini_midis/121437_1.mid"

md = MidiDict.from_midi(mid_path=mf)
print(len(md.note_msgs))
print(md.note_msgs[:10])
