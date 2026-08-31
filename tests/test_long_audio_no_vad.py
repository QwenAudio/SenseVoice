import io
import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from long_audio_no_vad import (
    create_sensevoice_model,
    ffmpeg_pcm_stream,
    iter_overlapping_windows,
    merge_transcripts,
    write_outputs_atomically,
)


class ShortReadStream(io.BytesIO):
    def read(self, size=-1):
        if size < 0:
            return super().read(size)
        return super().read(min(size, 5))


class LongAudioNoVadTest(unittest.TestCase):
    def test_model_is_created_without_vad_or_remote_code(self):
        calls = []

        class FakeAutoModel:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        create_sensevoice_model(
            FakeAutoModel,
            model_name="iic/SenseVoiceSmall",
            device="cpu",
        )

        self.assertEqual(
            calls,
            [
                {
                    "model": "iic/SenseVoiceSmall",
                    "device": "cpu",
                    "disable_update": True,
                }
            ],
        )

    def test_windows_cover_input_and_do_not_emit_overlap_only_tail(self):
        samples = np.arange(11, dtype=np.int16)
        stream = ShortReadStream(samples.tobytes())

        windows = list(
            iter_overlapping_windows(
                stream,
                sample_rate=1,
                window_seconds=6,
                overlap_seconds=2,
            )
        )

        self.assertEqual([start for start, _ in windows], [0, 4, 8])
        self.assertEqual(
            [window.tolist() for _, window in windows],
            [[0, 1, 2, 3, 4, 5], [4, 5, 6, 7, 8, 9], [8, 9, 10]],
        )

    def test_window_validation_rejects_invalid_overlap(self):
        for overlap in (0, 6, 7):
            with self.subTest(overlap=overlap):
                with self.assertRaisesRegex(ValueError, "overlap_seconds"):
                    list(
                        iter_overlapping_windows(
                            io.BytesIO(b""),
                            sample_rate=1,
                            window_seconds=6,
                            overlap_seconds=overlap,
                        )
                    )

    def test_merge_only_removes_exact_boundary_overlap(self):
        self.assertEqual(
            merge_transcripts(["alpha beta", "beta gamma"], min_overlap_chars=4),
            "alpha beta gamma",
        )
        self.assertEqual(
            merge_transcripts(["repeat", "unrelated"], min_overlap_chars=4),
            "repeat unrelated",
        )

    def test_atomic_outputs_keep_raw_chunks_when_merged_text_deduplicates(self):
        chunks = [
            {"index": 0, "start_ms": 0, "end_ms": 6000, "raw_text": "alpha beta"},
            {"index": 1, "start_ms": 4000, "end_ms": 10000, "raw_text": "beta gamma"},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            text_path = Path(tmp) / "result.txt"
            jsonl_path = Path(tmp) / "result.chunks.jsonl"
            write_outputs_atomically(
                text_path,
                jsonl_path,
                "alpha beta gamma",
                chunks,
            )

            self.assertEqual(text_path.read_text(encoding="utf-8"), "alpha beta gamma\n")
            records = [
                json.loads(line)
                for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(records, chunks)

    def test_ffmpeg_stream_decodes_container_audio_to_mono_16khz_pcm(self):
        samples = np.array([0, 1000, -1000, 2000], dtype=np.int16)
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "input.wav"
            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(samples.tobytes())

            with ffmpeg_pcm_stream(wav_path) as stream:
                decoded = np.frombuffer(stream.read(), dtype="<i2")

            np.testing.assert_array_equal(decoded, samples)


if __name__ == "__main__":
    unittest.main()
