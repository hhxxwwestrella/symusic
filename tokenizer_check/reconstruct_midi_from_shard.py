#!/usr/bin/env python3
import io, json, tarfile, glob, os
from pathlib import Path
import numpy as np
import mido, pretty_midi

from ariautils.tokenizer.absolute import AbsTokenizer
from ariautils.midi import MidiDict  # only for type clarity; not strictly required

# If your ariautils exposes these, we’ll use them; we fall back gracefully otherwise.
def _mididict_from_events_or_msgdict(obj):
    """Accepts either event stream (from tok.decode) or msg-dict (from tok.detokenize)."""
    from ariautils.midi import MidiDict
    if isinstance(obj, dict):
        if hasattr(MidiDict, "from_msg_dict"):
            return MidiDict.from_msg_dict(obj)
    else:
        if hasattr(MidiDict, "from_events"):
            return MidiDict.from_events(obj)
        if hasattr(MidiDict, "from_event_stream"):
            return MidiDict.from_event_stream(obj)

def _mididict_to_mido(mididict):
    """Convert MidiDict -> mido.MidiFile using whatever is available in your repo."""
    # Preferred function name:
    try:
        from ariautils.midi import dict_to_midi   # common name
        return dict_to_midi(mididict)
    except Exception:
        pass
    # Alternate function name:
    try:
        from ariautils.midi import midi_dict_to_mido
        return midi_dict_to_mido(mididict)
    except Exception:
        pass
    # Instance method?
    if hasattr(mididict, "to_mido"):
        return mididict.to_mido()
    # PrettyMIDI route as a last resort:
    if hasattr(mididict, "to_pretty_midi"):
        pm = mididict.to_pretty_midi()
        buf = io.BytesIO()
        pm.write(buf); buf.seek(0)
        return mido.MidiFile(file=buf)
    raise RuntimeError("No MidiDict -> mido conversion path found.")

def tokens_to_mido(ids):
    tok = AbsTokenizer()
    # Try decode → events first
    events = None
    if hasattr(tok, "decode"):
        try:
            events = tok.decode(list(map(int, ids)))
        except Exception:
            events = None
    if events is not None:
        md = _mididict_from_events_or_msgdict(events)
        return _mididict_to_mido(md)

    # Try detokenize → msg dict
    if hasattr(tok, "detokenize"):
        msg_dict = tok.detokenize(list(map(int, ids)))
        md = _mididict_from_events_or_msgdict(msg_dict)
        return _mididict_to_mido(md)

    # If your tokenizer has a direct helper:
    if hasattr(tok, "decode_to_mididict"):
        md = tok.decode_to_mididict(list(map(int, ids)))
        return _mididict_to_mido(md)

    raise RuntimeError("Tokenizer lacks decode/detokenize APIs needed for reconstruction.")

def load_downbeats(pm: pretty_midi.PrettyMIDI):
    try:
        db = pm.get_downbeats()
        if db is not None and len(db) >= 2:
            return np.asarray(db, dtype=np.float32)
    except Exception:
        pass
    # fallback 4/4 @ 120 BPM
    total = pm.get_end_time()
    if total <= 0:
        return np.array([0.0, 1.0], dtype=np.float32)
    bar_sec = 2.0
    n_bars = max(2, int(np.ceil(total / bar_sec)) + 1)
    return (np.arange(n_bars, dtype=np.float32) * bar_sec)

def copy_tempo_and_meter(src_pm: pretty_midi.PrettyMIDI, dst_mid: mido.MidiFile, start_t: float):
    # tempo at start_t
    try:
        times, bpms = src_pm.get_tempo_changes()
        if len(bpms):
            import bisect
            idx = max(0, bisect.bisect_right(times, start_t) - 1)
            bpm = float(bpms[idx])
        else:
            bpm = 120.0
    except Exception:
        bpm = 120.0
    # meter at start_t
    try:
        tsigs = src_pm.time_signature_changes
        if tsigs:
            times = [ts.time for ts in tsigs]
            import bisect
            idx = max(0, bisect.bisect_right(times, start_t) - 1)
            num, den = tsigs[idx].numerator, tsigs[idx].denominator
        else:
            num, den = 4, 4
    except Exception:
        num, den = 4, 4

    if not dst_mid.tracks:
        dst_mid.tracks.append(mido.MidiTrack())
    track0 = dst_mid.tracks[0]
    # insert at time=0
    track0.insert(0, mido.MetaMessage("time_signature", numerator=num, denominator=den, time=0))
    mpb = int(60_000_000 / max(1e-6, bpm))
    track0.insert(1, mido.MetaMessage("set_tempo", tempo=mpb, time=0))

def open_first_shard(dir_path: str | Path) -> Path:
    candidates = sorted(glob.glob(os.path.join(str(dir_path), "tokens-*.tar*")))
    if not candidates:
        raise FileNotFoundError("No shard files found (tokens-*.tar*)")
    return Path(candidates[0])

def extract_item(shard_path: Path, uid: str):
    with tarfile.open(shard_path, "r:*") as tf:
        base = None
        # meta is stored at "<uid>.meta.json"
        for m in tf.getmembers():
            if m.name.endswith(".meta.json") and m.name.split("/")[-1].startswith(uid):
                base = m.name[:-len(".meta.json")]
                meta = json.load(io.BytesIO(tf.extractfile(m).read()))
                break
        if base is None:
            raise KeyError(f"UID {uid} not found in shard {shard_path.name}")

        def load_npy(suffix):
            name = f"{base}.{suffix}"
            f = tf.extractfile(name)
            return np.load(io.BytesIO(f.read()))
        phrase_ids  = load_npy("phrase_ids.npy")
        phrase_attn = load_npy("phrase_attn.npy")
        context_ids = load_npy("context_ids.npy")
        context_attn= load_npy("context_attn.npy")
        return meta, phrase_ids, context_ids

def ids_to_mido(ids, attn=None):
    """
    Convert encoded ids (+ optional attention mask) back to a mido.MidiFile.
    Uses AbsTokenizer.id_to_tok and detokenize(tokens) -> MidiDict -> to_midi().
    """
    tok = AbsTokenizer()

    # 1) filter out pads using attention if provided; otherwise drop pad_id explicitly
    if attn is not None and len(attn) == len(ids):
        active_ids = [int(i) for i, m in zip(ids, attn) if int(m) != 0]
    else:
        active_ids = [int(i) for i in ids if int(i) != tok.pad_id]

    # 2) map ids -> tokens
    try:
        tokens = [tok.id_to_tok[i] for i in active_ids]
    except Exception:
        # helpful debug
        bad = [i for i in active_ids if i < 0 or i >= tok.vocab_size]
        raise ValueError(f"Found ids out of vocab: {bad[:10]} (showing up to 10)")

    # 3) detokenize tokens -> MidiDict
    md: MidiDict = tok.detokenize(tokens)  # this returns a MidiDict in your repo

    # 4) MidiDict -> mido.MidiFile
    mid = md.to_midi()
    return mid

if __name__ == "__main__":
    # ---- Fill these from your printout ----
    SHARDS_DIR = "./shards"   # directory that contains tokens-00000.tar.gz
    UID = "4c61149a0ddc5fc313208faf4ac695778469a69368c7ad376cb99c45b526f480"
    SRC_MIDI = "../mirex2025/sym-music-gen/data/filtered_aria/train/100315_0.mid"
    PHRASE_WINDOW = (76.0, 84.0)  # (start_s, end_s)
    CONTEXT_WINDOW = (52.0, 76.0)

    shard = open_first_shard(SHARDS_DIR)
    meta, phrase_ids, context_ids = extract_item(shard, UID)

    # Decode tokens -> Midi
    mid_phrase = ids_to_mido(phrase_ids, attn=None)
    mid_context = ids_to_mido(context_ids, attn=None)

    # Reapply tempo/meter from source section start
    src_pm = pretty_midi.PrettyMIDI(SRC_MIDI)
    copy_tempo_and_meter(src_pm, mid_phrase, PHRASE_WINDOW[0])
    copy_tempo_and_meter(src_pm, mid_context, CONTEXT_WINDOW[0])

    # Save reconstructed files
    mid_phrase.save("recon_phrase.mid")
    mid_context.save("recon_context.mid")

    # Optional: quick timing sanity (compare durations in seconds)
    def mido_seconds(mid: mido.MidiFile):
        # naive estimate via pretty_midi
        buf = io.BytesIO(); mid.save(file=buf); buf.seek(0)
        return pretty_midi.PrettyMIDI(buf).get_end_time()
    print("Reconstructed phrase len (s): ", mido_seconds(mid_phrase))
    print("Reconstructed context len (s):", mido_seconds(mid_context))
    print("Wrote recon_phrase.mid and recon_context.mid")
