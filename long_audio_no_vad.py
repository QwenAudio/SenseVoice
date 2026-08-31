#!/usr/bin/env python3
"""Transcribe long audio with SenseVoice using bounded, overlapping windows."""

import argparse
import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def create_sensevoice_model(auto_model_class, *, model_name, device):
    """Build the integrated SenseVoice model without adding a VAD pipeline."""
    return auto_model_class(
        model=model_name,
        device=device,
        disable_update=True,
    )


@contextmanager
def ffmpeg_pcm_stream(input_path):
    """Yield ffmpeg stdout as mono 16 kHz little-endian int16 PCM."""
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "pipe:1",
    ]
    with tempfile.TemporaryFile() as errors:
        try:
            decoder = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        except FileNotFoundError as error:
            raise RuntimeError("ffmpeg is required but was not found on PATH") from error

        try:
            yield decoder.stdout
        except BaseException:
            decoder.terminate()
            decoder.wait()
            raise
        finally:
            if decoder.stdout:
                decoder.stdout.close()

        return_code = decoder.wait()
        if return_code:
            errors.seek(0)
            detail = errors.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {detail}")


def iter_overlapping_windows(
    stream,
    *,
    sample_rate=16000,
    window_seconds=30,
    overlap_seconds=2,
):
    """Yield ``(start_sample, pcm)`` from a little-endian int16 stream."""
    if sample_rate <= 0 or window_seconds <= 0:
        raise ValueError("sample_rate and window_seconds must be positive")
    if not 0 < overlap_seconds < window_seconds:
        raise ValueError("overlap_seconds must be greater than 0 and below window_seconds")

    window_samples = int(sample_rate * window_seconds)
    overlap_samples = int(sample_rate * overlap_seconds)
    if overlap_samples <= 0 or overlap_samples >= window_samples:
        raise ValueError("overlap_seconds produces an invalid sample count")

    stride_samples = window_samples - overlap_samples
    window_bytes = window_samples * 2
    stride_bytes = stride_samples * 2
    buffer = bytearray()
    start_sample = 0
    furthest_emitted_sample = 0
    eof = False

    while True:
        while len(buffer) < window_bytes and not eof:
            block = stream.read(window_bytes - len(buffer))
            if block:
                buffer.extend(block)
            else:
                eof = True

        if len(buffer) % 2:
            raise RuntimeError("decoder returned an incomplete int16 PCM sample")

        available_samples = min(len(buffer) // 2, window_samples)
        end_sample = start_sample + available_samples
        if available_samples == 0 or end_sample <= furthest_emitted_sample:
            break

        pcm = np.frombuffer(bytes(buffer[: available_samples * 2]), dtype="<i2").copy()
        yield start_sample, pcm
        furthest_emitted_sample = end_sample

        if len(buffer) <= stride_bytes:
            buffer.clear()
        else:
            del buffer[:stride_bytes]
        start_sample += stride_samples


def merge_transcripts(texts, *, min_overlap_chars=4, max_overlap_chars=200):
    """Merge chunks, removing only exact case-insensitive suffix/prefix overlap."""
    merged = ""
    for text in texts:
        current = text.strip()
        if not current:
            continue
        if not merged:
            merged = current
            continue

        limit = min(len(merged), len(current), max_overlap_chars)
        overlap = 0
        for size in range(limit, min_overlap_chars - 1, -1):
            if merged[-size:].casefold() == current[:size].casefold():
                overlap = size
                break

        remainder = current[overlap:]
        if not remainder:
            continue
        separator = "" if merged[-1:].isspace() or remainder[:1].isspace() else " "
        merged += separator + remainder
    return merged


def write_outputs_atomically(text_path, jsonl_path, merged_text, chunks):
    """Publish merged text and lossless chunk records only after both serialize."""
    text_path = Path(text_path)
    jsonl_path = Path(jsonl_path)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    text_tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=text_path.parent, delete=False
    )
    jsonl_tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=jsonl_path.parent, delete=False
    )
    try:
        with text_tmp, jsonl_tmp:
            text_tmp.write(merged_text + "\n")
            for chunk in chunks:
                json.dump(chunk, jsonl_tmp, ensure_ascii=False)
                jsonl_tmp.write("\n")
        os.replace(jsonl_tmp.name, jsonl_path)
        os.replace(text_tmp.name, text_path)
    finally:
        for name in (text_tmp.name, jsonl_tmp.name):
            if os.path.exists(name):
                os.unlink(name)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="SenseVoice long-audio inference without VAD"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("transcript.txt"))
    parser.add_argument("--chunks-output", type=Path)
    parser.add_argument("--model", default="iic/SenseVoiceSmall")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--window-seconds", type=float, default=30)
    parser.add_argument("--overlap-seconds", type=float, default=2)
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--no-itn", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    model = create_sensevoice_model(
        AutoModel,
        model_name=args.model,
        device=args.device,
    )
    chunks = []
    try:
        with ffmpeg_pcm_stream(args.input) as pcm_stream:
            for index, (start_sample, pcm_i16) in enumerate(
                iter_overlapping_windows(
                    pcm_stream,
                    window_seconds=args.window_seconds,
                    overlap_seconds=args.overlap_seconds,
                )
            ):
                pcm = pcm_i16.astype(np.float32) / 32768.0
                result = model.generate(
                    input=pcm,
                    cache={},
                    language=args.language,
                    use_itn=not args.no_itn,
                    batch_size_s=args.window_seconds,
                )
                raw_text = result[0]["text"]
                text = rich_transcription_postprocess(raw_text)
                chunks.append(
                    {
                        "index": index,
                        "start_ms": round(start_sample * 1000 / 16000),
                        "end_ms": round((start_sample + len(pcm_i16)) * 1000 / 16000),
                        "raw_text": raw_text,
                        "text": text,
                    }
                )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    texts = [chunk["text"] for chunk in chunks]
    merged = " ".join(texts) if args.no_dedupe else merge_transcripts(texts)
    chunks_output = args.chunks_output or args.output.with_suffix(".chunks.jsonl")
    write_outputs_atomically(args.output, chunks_output, merged, chunks)
    print(f"wrote {args.output} and {chunks_output} ({len(chunks)} windows)")


if __name__ == "__main__":
    main()
