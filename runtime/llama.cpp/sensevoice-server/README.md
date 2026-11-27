# sensevoice-server — OpenAI-compatible STT server (SenseVoiceSmall on ggml)

Streaming + file transcription over HTTP, backed by the **SenseVoiceSmall** ggml
runtime with built-in **FSMN-VAD**. One self-contained C++ binary, CPU-only, no
Python at runtime.

Provides two OpenAI-compatible surfaces:

- **REST** `POST /v1/audio/transcriptions` — transcribe an uploaded audio file
  (any format decodable by miniaudio → 16k mono) in `json` / `text` /
  `verbose_json` / `srt` / `vtt`, or as an SSE stream.
- **Realtime WebSocket** `/v1/realtime?intent=transcription` — stream base64
  PCM16 chunks and receive incremental **partial** transcripts plus VAD
  end-pointed **final** transcripts (OpenAI realtime transcription protocol).

## Running it

```bash
sensevoice-server -m model/sensevoice-small-q8.gguf -vad model/fsmn-vad.gguf [host [port]]
```

- `-m, --model`    SenseVoice GGUF (required)
- `-vad, --vad`    FSMN-VAD GGUF — used for sentence/utterance end-pointing
- `-ngl, --ngl <N>`  layers offloaded to CUDA GPU; any value > 0 runs the whole model
  on device 0 (default `0` = CPU). Requires a build with `-DGGML_CUDA=ON`; the server
  falls back to CPU when no CUDA device is present. The fbank frontend and FSMN-VAD
  always stay on CPU.
- `--keep-tags`    keep `<|lang|>/<|emo|>/<|event|>` meta tokens in text
- `--threads <N>`  ggml compute threads (default 8)
- `--partial-ms <N>`  WS partial-transcription cadence, ms (default 400)
- `--vad-maxseg <ms>` max single VAD segment (default 30000)
- `--vad-slot-ms <ms>`  idle slot: when no audible chunk arrives within this window the
  VAD buffers are finalized and reset until new chunks come in (default 2000).
  Prevents a connected-but-idle session (mic live, nobody talking) from keeping the
  VAD busy re-scoring quiet chunks.
- `--read-timeout-s <N>` socket read timeout, s (default 300)
- `--max-connections <N>` maximum active WebSocket clients (default 4;
  accepted range 1 to 128)
- `--max-audio-seconds <N>` maximum decoded audio per REST request and maximum
  buffered audio per WebSocket session (default 300; accepted range 1 to 3600)
- `--web <dir>`      serve the directory at `/` (mic webui; open `http://host:port/`)
- positional `host` (default `127.0.0.1`) and `port` (default `8040`)

```
[server] listening on 127.0.0.1:8040
[server]   POST /v1/audio/transcriptions
[server]   WS   /v1/realtime?intent=transcription
```

## Build

Same build as the CLI tools (fetches pinned llama.cpp; static, self-contained):

```bash
cd runtime/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release     # -> build/bin/sensevoice-server
cmake --build build -j
```

## REST: file transcription

Upload any audio file (wav/mp3/flac/ogg/...). FSMN-VAD segments it internally;
each segment is transcribed and results are concatenated.

```bash
# plain JSON (default)
curl -F file=@audio.wav http://127.0.0.1:8040/v1/audio/transcriptions
# {"text":"我想问我在滨海新区有房。"}

# plain text
curl -F file=@audio.wav -F response_format=text http://127.0.0.1:8040/v1/audio/transcriptions

# segments + timestamps
curl -F file=@audio.wav -F response_format=verbose_json http://127.0.0.1:8040/v1/audio/transcriptions
# {"task":"transcribe","language":"zh","duration":6.01,"text":"...",
#  "segments":[{"id":0,"start":0.77,"end":5.98,"start_ms":770,"end_ms":5980,"text":"..."}]}

# subtitles
curl -F file=@audio.wav -F response_format=srt  http://127.0.0.1:8040/v1/audio/transcriptions
curl -F file=@audio.wav -F response_format=vtt  http://127.0.0.1:8040/v1/audio/transcriptions

# SSE stream: transcript.text.delta, then transcript.text.done
curl -N -F file=@audio.wav -F stream=true http://127.0.0.1:8040/v1/audio/transcriptions
```

Other OpenAI-compatible endpoints: `GET /v1/models` (model list) and
`GET /health` (liveness).

## Web UI — live mic transcription

A zero-build browser UI (`webui/index.html`) streams the microphone to the
realtime endpoint and renders dictation live:

```bash
sensevoice-server -m model/sensevoice-small-q8.gguf -vad model/fsmn-vad.gguf \
    --web runtime/llama.cpp/sensevoice-server/webui
# open http://127.0.0.1:8040/
```

- Mic audio is captured in the browser, downsampled to 16 kHz mono, base64-PCM16
  encoded, and sent as `input_audio_buffer.append` chunks.
- While a sentence is still open (interim), the current line **updates in
  place** from `...transcription.delta` messages.
- When VAD commits the sentence, `...transcription.completed` moves it to its
  **own line** and a fresh interim line starts.
- Top bar shows WS / VAD / mic status plus an input-level meter; the host/port
  field lets you point the page at a server running elsewhere (no `--web`
  required).

## Realtime WebSocket: streaming transcription

Client protocol (OpenAI realtime transcription):

| direction     | event                                                        |
|---------------|--------------------------------------------------------------|
| client → server | `session.update` (set `turn_detection.type: "server_vad"` or `"none"`) |
| client → server | `input_audio_buffer.append`  — `{"audio": "<base64 pcm16>"}`  (16 kHz mono) |
| client → server | `input_audio_buffer.commit` — force-endpoint the current turn |
| client → server | `input_audio_buffer.clear`  — reset the session buffer      |
| server → client | `session.created`                                            |
| server → client | `input_audio_buffer.speech_started/stopped` (VAD turn detection) |
| server → client | `conversation.item.input_audio_transcription.delta` — partial with current `transcript` |
| server → client | `input_audio_buffer.committed`, `conversation.item.created` |
| server → client | `conversation.item.input_audio_transcription.completed` — final `transcript` |

The audio buffer is **16 kHz mono PCM16**; base64-encode every chunk and stream
it via `input_audio_buffer.append`. FSMN-VAD runs incrementally on each chunk:
it end-points utterances, emits `speech_started`/`speech_stopped`, and streams
hypothesis updates (throttled by `--partial-ms`) while speech is open. For each
`...transcription.delta`, replace the current partial for its `item_id` with
`transcript`. An empty `transcript` clears the partial. The `delta` field keeps
its legacy behavior, with a suffix on extension and the revised text on a
rewrite. On segment finalization, the final `transcript` arrives in
`...transcription.completed`.

Idle CPU: VAD only scores frames when new audio actually arrives. Chunks with no
real signal are written straight into the score cache as silence votes without
running the NN, and once `--vad-slot-ms` elapses with no audible chunk the open
VAD state and session buffer are reset, so a connected-but-quiet client parks at
~0% CPU until the next speech burst.

Try it with the bundled client (zero-dependency, stdlib socket RFC6455):

```bash
# stream a 16k mono PCM16 WAV in 200 ms chunks; prints each completed transcript
python3 runtime/llama.cpp/tests/stream_client.py 127.0.0.1 8040 audio_16k.wav 200
# Oh sugar, you can run, you can taste, but weep both no, you love the thrill, so why not give in.
```

## Implementation notes

- The SenseVoice engine (80-mel fbank + LFR, SAN-M encoder, CTC head, detok) is
  duplicated verbatim from the validated `funasr-sensevoice` CLI so the CLI and
  its goldens stay untouched.
- Streaming VAD recomputes the whole-buffer FSMN-VAD forward per poll and steps
  a faithful copy of the `funasr_vad.h` / E2EVadModel state machine — segment
  boundaries match the whole-file VAD tool (e.g. `770 5980` on `sample.wav`).
- Transport is vendored cpp-httplib (WebSocket, multipart, SSE/`DataSink`).
  Long-lived WebSockets use a bounded thread pool. `--max-connections` caps
  active WebSocket sessions while two workers remain available for HTTP.
- REST uploads are decoded to 16k mono by miniaudio (any sample rate / channel
  count / bit depth).

## Tests

```bash
# end-to-end: REST json/text/verbose_json/SSE + WS streaming vs the golden transcript
MODEL_GGUF=model/sensevoice-small-q8.gguf VAD_GGUF=model/fsmn-vad.gguf \
    runtime/llama.cpp/tests/run_server_smoke.sh
```

`tests/stream_client.py` is the WS test client; `tests/run_regression.sh` also
runs the server smoke when the binary and models are present.
