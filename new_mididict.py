from ariautils.midi import *
import io

class MidiDict_allowing_buffer(MidiDict):
    def __init__(
            self,
            meta_msgs: list[MetaMessage],
            tempo_msgs: list[TempoMessage],
            pedal_msgs: list[PedalMessage],
            instrument_msgs: list[InstrumentMessage],
            note_msgs: list[NoteMessage],
            ticks_per_beat: int,
            metadata: dict[str, Any],
    ):
        super().__init__(
            meta_msgs=meta_msgs,
            tempo_msgs=tempo_msgs,
            pedal_msgs=pedal_msgs,
            instrument_msgs=instrument_msgs,
            note_msgs=note_msgs,
            ticks_per_beat=ticks_per_beat,
            metadata=metadata,
        )

    @classmethod
    def from_midi_bytes(cls, mid_bytes: bytes) -> "MidiDict_allowing_buffer":
        """Loads a MIDI file from an in-memory byte buffer and returns subclass."""
        mid = mido.MidiFile(file=io.BytesIO(mid_bytes))
        midi_dict = cls(**midi_to_dict(mid))
        midi_dict.metadata["abs_load_path"] = "<in-memory>"
        return midi_dict
