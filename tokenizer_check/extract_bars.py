#!/usr/bin/env python3
import argparse, math, io, bisect
from pathlib import Path
import numpy as np
import mido, pretty_midi

def load_downbeats(pm: pretty_midi.PrettyMIDI) -> np.ndarray:
    try:
        db = pm.get_downbeats()
        if db is not None and len(db) >= 2:
            return np.asarray(db, dtype=np.float32)
    except Exception:
        pass
    # Fallback: 4/4 at ~120 BPM (2.0s/bar)
    total = pm.get_end_time()
    if total <= 0:  # degenerate
        return np.array([0.0, 1.0], dtype=np.float32)
    bar_sec = 2.0
    n_bars = max(2, int(math.ceil(total / bar_sec)) + 1)
    return np.arange(n_bars, dtype=np.float32) * bar_sec

def slice_pretty_midi(pm: pretty_midi.PrettyMIDI, start_t: float, end_t: float) -> pretty_midi.PrettyMIDI:
    out = pretty_midi.PrettyMIDI()
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        new_inst = pretty_midi.Instrument(program=inst.program, is_drum=False, name=inst.name)
        for n in inst.notes:
            if n.end <= start_t or n.start >= end_t:
                continue
            new_inst.notes.append(pretty_midi.Note(
                velocity=n.velocity,
                pitch=n.pitch,
                start=max(n.start, start_t) - start_t,
                end=min(n.end, end_t) - start_t,
            ))
        if new_inst.notes:
            out.instruments.append(new_inst)
    return out

def copy_tempo_and_meter(src_pm: pretty_midi.PrettyMIDI, dst_mid: mido.MidiFile, start_t: float):
    # tempo at start_t
    try:
        times, bpms = src_pm.get_tempo_changes()
        if len(bpms):
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
            ts_times = [ts.time for ts in tsigs]
            idx = max(0, bisect.bisect_right(ts_times, start_t) - 1)
            num, den = tsigs[idx].numerator, tsigs[idx].denominator
        else:
            num, den = 4, 4
    except Exception:
        num, den = 4, 4

    if not dst_mid.tracks:
        dst_mid.tracks.append(mido.MidiTrack())
    track0 = dst_mid.tracks[0]
    # insert meta at time 0
    track0.insert(0, mido.MetaMessage('time_signature', numerator=num, denominator=den, time=0))
    mpb = int(60_000_000 / max(1e-6, bpm))
    track0.insert(1, mido.MetaMessage('set_tempo', tempo=mpb, time=0))

def pm_to_mido(pm: pretty_midi.PrettyMIDI) -> mido.MidiFile:
    buf = io.BytesIO()
    pm.write(buf); buf.seek(0)
    return mido.MidiFile(file=buf)

def main():
    ap = argparse.ArgumentParser(description="Extract a bar range from a MIDI using downbeats.")
    ap.add_argument("--midi", required=True, help="Path to source .mid")
    ap.add_argument("--start-bar", type=int, required=True, help="First bar index (0-based)")
    ap.add_argument("--end-bar", type=int, required=True, help="End bar index (exclusive)")
    ap.add_argument("--out", default="orig_phrase.mid", help="Output .mid path")
    args = ap.parse_args()

    pm = pretty_midi.PrettyMIDI(args.midi)
    downbeats = load_downbeats(pm)

    if args.start_bar < 0 or args.end_bar > len(downbeats)-1 or args.start_bar >= args.end_bar:
        raise ValueError(f"Bar range [{args.start_bar}, {args.end_bar}) out of bounds for {len(downbeats)-1} bars")

    start_t = float(downbeats[args.start_bar])
    end_t   = float(downbeats[args.end_bar])

    sliced_pm = slice_pretty_midi(pm, start_t, end_t)
    mid = pm_to_mido(sliced_pm)
    copy_tempo_and_meter(pm, mid, start_t)
    mid.save(args.out)

    print(f"Wrote {args.out}  (bars {args.start_bar}–{args.end_bar-1}, times {start_t:.3f}–{end_t:.3f}s)")

if __name__ == "__main__":
    main()
