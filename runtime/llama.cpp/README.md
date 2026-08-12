# SenseVoiceSmall on llama.cpp / GGUF

Run **SenseVoiceSmall** on the [llama.cpp](https://github.com/ggml-org/llama.cpp)
/ ggml stack — **CPU, edge, a single binary, no Python at runtime**. Like
[whisper.cpp](https://github.com/ggml-org/whisper.cpp), but for SenseVoice.

## Why this exists

SenseVoiceSmall normally runs on PyTorch / ONNX / libtorch. This runtime ports it
to **ggml + GGUF** so it can run CPU-only, offline, embedded in a C/C++ app, with
quantized weights. Use it on laptops / phones / edge boxes where there is no GPU
and no Python. (For high-QPS GPU serving, the PyTorch/vLLM path is still the way.)

## Architecture

SenseVoiceSmall = **SAN-M encoder (70 layers) + CTC head** — no LLM, no autoregression.
The whole pipeline runs in C++:

```
 audio.wav (16k mono)
      │  kaldi 80-mel fbank + LFR                         (C++)
      ▼
   features [T, 560]
      │  prepend 4 query tokens [lang, event, emotion, itn]
      ▼
   [4 + T, 560]
      │  SAN-M encoder                                    (ggml)  ── sensevoice-small.gguf
      ▼
   encoder out [4+T, 512]
      │  CTC head (Linear 512→25055) → greedy CTC (argmax, dedup, drop blank)
      ▼
   token ids
      │  SentencePiece detok                             (detok.py)
      ▼
   <|zh|><|NEUTRAL|><|Speech|><|woitn|> transcription...
```

The SAN-M encoder is the same architecture as Fun-ASR-Nano's, so the ggml forward
is shared between the two runtimes.

## Download pre-built GGUF (fastest — no Python ML env)
```bash
./download-funasr-model.sh sensevoice          # pulls SenseVoice + fsmn-vad GGUF from Hugging Face
llama-funasr-sensevoice -m funasr-gguf/sensevoice-small-f16.gguf \
    -a audio.wav --vad funasr-gguf/fsmn-vad.gguf
```
Pre-converted GGUF: [FunAudioLLM/SenseVoiceSmall-GGUF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF) · [fsmn-vad-GGUF](https://huggingface.co/FunAudioLLM/fsmn-vad-GGUF). Or convert yourself: `python convert-funasr-to-gguf.py sensevoice --wtype f16`.

## Build (standalone, CI-friendly)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release      # fetches pinned llama.cpp; static, self-contained
cmake --build build -j                          # -> build/bin/llama-funasr-*
```
## Quickstart

**1. Build:**
```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cp -r /path/to/runtime/llama.cpp/funasr-common examples/   # shared audio loader (miniaudio); each example CMake adds ../funasr-common
cp -r /path/to/runtime/llama.cpp/funasr-sensevoice examples/
echo 'add_subdirectory(funasr-sensevoice)' >> examples/CMakeLists.txt
cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build build -j --target llama-funasr-sensevoice
```

**2. Convert weights** (needs the checkpoint, e.g. `FunAudioLLM/SenseVoiceSmall`):
```bash
python runtime/llama.cpp/export_sensevoice_gguf.py \
    --model_pt <model>/model.pt --mvn <model>/am.mvn \
    --out sensevoice-small.gguf                          # f32, ~936 MB
python runtime/llama.cpp/export_sensevoice_gguf.py --wtype f16 \
    --model_pt <model>/model.pt --mvn <model>/am.mvn \
    --out sensevoice-small-f16.gguf                       # half size
```

To make the GGUF directly loadable by audio.cpp's spec-backed runtime, pass a
`sense_asr` schema-v1 model specification during export:
```bash
python runtime/llama.cpp/export_sensevoice_gguf.py --wtype q8_0 \
    --model_pt <model>/model.pt --mvn <model>/am.mvn \
    --model-spec model_specs/sense_asr.json \
    --out sensevoice-small-q8-audiocpp-v1.gguf
```

**3. Transcribe:**
```bash
build/bin/llama-funasr-sensevoice -m sensevoice-small.gguf -a audio.wav   # prints transcription text
# --keep-tags keeps the <|lang|>/<|emotion|>/<|event|> tags; --ids prints raw CTC ids
```
Expected output:
```
我想问我在滨海新区有房我一直没有照顾孩子...你觉得这是正常的想法吗
```
The leading `<|...|>` tags are the predicted language / emotion / event / ITN.

**4. Long audio — built-in FSMN-VAD (no Python front end):**
```bash
python runtime/llama.cpp/export_vad_gguf.py \
    --model_pt <fsmn-vad>/model.pt --mvn <fsmn-vad>/am.mvn --out fsmn-vad.gguf
build/bin/llama-funasr-sensevoice -m sensevoice-small.gguf -a long.wav \
    --vad fsmn-vad.gguf          # segments internally, then concatenates
```
The runtime then needs **no Python for segmentation** — it runs FSMN-VAD (native ggml)
inside the binary and decodes each speech segment. Segment boundaries match the PyTorch
`fsmn-vad` front end within ~10 ms; see [BENCHMARKS.md](BENCHMARKS.md) for full-184 numbers.

## Accuracy & validation

- **CTC token ids (C++) vs PyTorch:** **identical** (108/108 on a benchmark clip).
- **Detokenized text:** matches the FunASR `AutoModel` output **exactly**.
- Encoder validated against PyTorch (shared with Fun-ASR-Nano runtime): cosine 1.0.
- Encode time ≈ **1.3 s** on CPU for a 44 s clip.

## Tips & gotchas

- **No CMVN at inference.** SenseVoice `inference()` feeds the **raw** log-mel fbank
  to the encoder; it does **not** apply `am.mvn`. Applying CMVN makes the model
  predict `<|nospeech|>`. (The export script reads `am.mvn` for completeness but the
  runtime does not use it.)
- **Query tokens (4)** are prepended from `embed.weight`, default indices
  `[language=auto(0), event=1, emotion=2, textnorm=woitn(15)]`. Change them for a
  fixed language or to enable ITN (`withitn=14`).
- **WAV input** assumes 16 kHz mono PCM16.
- LayerNorm eps = 1e-5; FSMN = exact f32 shift-accumulate; fbank matches torchaudio.

## Files
```
funasr-sensevoice/        ggml runtime: WAV → CTC token ids
export_sensevoice_gguf.py export encoder + CTC head + query embeddings to GGUF
detok.py                  SentencePiece id → text (bpe model ships with the checkpoint)
funasr-vad/               built-in FSMN-VAD tool + --vad library (funasr-common/funasr_vad.h)
export_vad_gguf.py        export FSMN-VAD encoder + CMVN to GGUF
```

## Roadmap
- Built-in SentencePiece detok (drop the Python step); arbitrary WAV formats;
  encoder Q8 quantization; timestamps.

## Further reading

See [DESIGN.md](DESIGN.md) for the full system design — architecture, the shared SAN-M encoder, GGUF weight format, numerical-fidelity and validation methodology, design trade-offs, and gotchas.
