([简体中文](./README_zh.md)|English|[日本語](./README_ja.md))


# Introduction

SenseVoice is a speech foundation model with multiple speech understanding capabilities, including automatic speech recognition (ASR),  spoken language identification (LID), speech emotion recognition (SER), and audio event detection (AED). 

<div align="center">  
<img src="image/sensevoice2.png">
</div>

[//]: # (<div align="center"><img src="image/sensevoice.png" width="700"/> </div>)

<div align="center">  
<h4>
<a href="https://www.funasr.com/en/"> Homepage </a>
｜<a href="#What's News"> What's News </a>
｜<a href="#Benchmarks"> Benchmarks </a>
｜<a href="#Install"> Install </a>
｜<a href="#Usage"> Usage </a>
｜<a href="#Community"> Community </a>
</h4>

Model Zoo:
[modelscope](https://www.modelscope.cn/models/iic/SenseVoiceSmall), [huggingface](https://huggingface.co/FunAudioLLM/SenseVoiceSmall), [paper](https://arxiv.org/abs/2407.04051)

Online Demo:
[modelscope demo](https://www.modelscope.cn/studios/iic/SenseVoice), [huggingface space](https://huggingface.co/spaces/FunAudioLLM/SenseVoice)


</div>

> **Released checkpoint scope:** SenseVoiceSmall supports ASR and language ID for Mandarin, Cantonese, English, Japanese, and Korean, together with emotion and audio-event tags. Speaker diarization is a composed FunASR pipeline using separate FSMN-VAD and CAM++ models; it is not an output of the SenseVoiceSmall checkpoint itself.


<a name="Highligts"></a>
# Highlights 🎯
**SenseVoice** focuses on high-accuracy multilingual speech recognition, speech emotion recognition, and audio event detection.
- **Research scope vs released checkpoint:** The broader SenseVoice work reports training on more than 400,000 hours and support for more than 50 languages. The released SenseVoiceSmall checkpoint linked above supports Mandarin, Cantonese, English, Japanese, and Korean; the benchmark comparisons below are task- and language-specific.
- **Rich transcribe:** 
  - Possess excellent emotion recognition capabilities, achieving and surpassing the effectiveness of the current best emotion recognition models on test data.
  - Offer sound event detection capabilities, supporting the detection of various common human-computer interaction events such as bgm, applause, laughter, crying, coughing, and sneezing.
- **Efficient Inference:** SenseVoiceSmall uses a non-autoregressive end-to-end framework for low-latency inference; see the benchmark setup and comparison below.
- **Convenient Finetuning:** Provide convenient finetuning scripts and strategies, allowing users to easily address long-tail sample issues according to their business scenarios.
- **Service Deployment:** Offer service deployment pipeline,  supporting multi-concurrent requests, with client-side languages including Python, C++, HTML, Java, and C#, among others.

<a name="What's News"></a>
# What's New 🔥
- 2026/07: **FunASR 1.3.29 restores SenseVoice VAD segment timestamps** — when token timestamps and a punctuation model are unavailable, `sentence_timestamp=True` now returns every VAD region through `sentence_info`, so subtitle and clipping clients receive usable segment boundaries instead of an empty timeline. Install with `pip install -U "funasr==1.3.29"`. [Release notes](https://github.com/modelscope/FunASR/releases/tag/v1.3.29) · [PyPI](https://pypi.org/project/funasr/1.3.29/)
- 2026/07: **FunASR 1.3.27 adds detected-language metadata for SenseVoice** — the OpenAI-compatible endpoint now reports detected `zh`, `en`, `yue`, `ja`, or `ko` in `verbose_json.language`. Install with `pip install -U "funasr==1.3.27"`. [Release notes](https://github.com/modelscope/FunASR/releases/tag/v1.3.27) · [API guide](https://www.funasr.com/en/blog/funasr-v1-3-27-language-metadata-vllm-fallback.html) · [PyPI](https://pypi.org/project/funasr/1.3.27/)
- 2026/06: **SenseVoice on llama.cpp / GGUF** — run it on CPU/edge as a single self-contained binary (whisper.cpp-style), built-in VAD, no Python at runtime. The q8 model is only ~254 MB with the same accuracy. [runtime/llama.cpp/](./runtime/llama.cpp/) · [Releases](https://github.com/QwenAudio/SenseVoice/releases) · [GGUF on Hugging Face](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF)
- 2026/05: FunASR can compose SenseVoiceSmall with separate FSMN-VAD, CAM++, and punctuation models to produce per-sentence speaker labels. Diarization is not a native SenseVoiceSmall checkpoint output. Requires installing FunASR from source: `pip install git+https://github.com/modelscope/FunASR.git`
- 2024/11: Add support for timestamp based on the CTC alignment.
- 2024/7: Added Export Features for [ONNX](./demo_onnx.py) and [libtorch](./demo_libtorch.py), as well as Python Version Runtimes: [funasr-onnx-0.4.0](https://pypi.org/project/funasr-onnx/), [funasr-torch-0.1.1](https://pypi.org/project/funasr-torch/)
- 2024/7: The [SenseVoice-Small](https://www.modelscope.cn/models/iic/SenseVoiceSmall) voice understanding model is open-sourced, which offers high-precision multilingual speech recognition, emotion recognition, and audio event detection capabilities for Mandarin, Cantonese, English, Japanese, and Korean and leads to exceptionally low inference latency.  
- 2024/7: The CosyVoice for natural speech generation with multi-language, timbre, and emotion control. CosyVoice excels in multi-lingual voice generation, zero-shot voice generation, cross-lingual voice cloning, and instruction-following capabilities. [CosyVoice repo](https://github.com/QwenAudio/CosyVoice) and [CosyVoice space](https://www.modelscope.cn/studios/iic/CosyVoice-300M).
- 2024/7: [FunASR](https://github.com/modelscope/FunASR) is a fundamental speech recognition toolkit that offers a variety of features, including speech recognition (ASR), Voice Activity Detection (VAD), Punctuation Restoration, Language Models, Speaker Verification, Speaker Diarization and multi-talker ASR.

<a name="Benchmarks"></a>
# Benchmarks 📝

## Multilingual Speech Recognition
We compared the performance of multilingual speech recognition between SenseVoice and Whisper on open-source benchmark datasets, including AISHELL-1, AISHELL-2, Wenetspeech, LibriSpeech, and Common Voice. In terms of Chinese and Cantonese recognition, the SenseVoice-Small model has advantages.

<div align="center">  
<img src="image/asr_results1.png" width="400" /><img src="image/asr_results2.png" width="400" />
</div>

## Speech Emotion Recognition

Due to the current lack of widely-used benchmarks and methods for speech emotion recognition, we conducted evaluations across various metrics on multiple test sets and performed a comprehensive comparison with numerous results from recent benchmarks. The selected test sets encompass data in both Chinese and English, and include multiple styles such as performances, films, and natural conversations. Without finetuning on the target data, SenseVoice was able to achieve and exceed the performance of the current best speech emotion recognition models.

<div align="center">  
<img src="image/ser_table.png" width="1000" />
</div>

Furthermore, we compared multiple open-source speech emotion recognition models on the test sets, and the results indicate that the SenseVoice-Large model achieved the best performance on nearly all datasets, while the SenseVoice-Small model also surpassed other open-source models on the majority of the datasets.

<div align="center">  
<img src="image/ser_figure.png" width="500" />
</div>

## Audio Event Detection

Although trained exclusively on speech data, SenseVoice can still function as a standalone event detection model. We compared its performance on the environmental sound classification ESC-50 dataset against the widely used industry models BEATS and PANN. The SenseVoice model achieved commendable results on these tasks. However, due to limitations in training data and methodology, its event classification performance has some gaps compared to specialized AED models.

<div align="center">  
<img src="image/aed_figure.png" width="500" />
</div>

## Computational  Efficiency

In the benchmark setup shown below, SenseVoiceSmall uses a non-autoregressive end-to-end architecture and, at a similar parameter count, runs more than 5 times faster than Whisper-Small and 15 times faster than Whisper-Large.

<div align="center">  
<img src="image/inference.png" width="1000" />
</div>


# Requirements

```shell
pip install -r requirements.txt
```

SenseVoiceSmall examples and the composed FunASR diarization path require `funasr>=1.3.26`. If you installed this repository earlier, run `pip install -U "funasr>=1.3.26"` before retrying the demos.

<a name="Usage"></a>
# Usage

## Inference

Supports input of audio in any format and of any duration.

```python
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "iic/SenseVoiceSmall"


model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",    
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)

# en
res = model.generate(
    input=f"{model.model_path}/example/en.mp3",
    cache={},
    language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(res[0]["text"])
print(text)
```

<details><summary>Parameter Description (Click to Expand)</summary>

- `model_dir`: The name of the model, or the path to the model on the local disk.
- `trust_remote_code`:
  - When `True`, it means that the model's code implementation is loaded from `remote_code`, which specifies the exact location of the `model` code (for example, `model.py` in the current directory). It supports absolute paths, relative paths, and network URLs.
  - When `False`, it indicates that the model's code implementation is the integrated version within [FunASR](https://github.com/modelscope/FunASR). At this time, modifications made to `model.py` in the current directory will not be effective, as the version loaded is the internal one from FunASR. For the model code, [click here to view](https://github.com/modelscope/FunASR/tree/main/funasr/models/sense_voice).
- `vad_model`: This indicates the activation of VAD (Voice Activity Detection). The purpose of VAD is to split long audio into shorter clips. In this case, the inference time includes both VAD and SenseVoice total consumption, and represents the end-to-end latency. If you wish to test the SenseVoice model's inference time separately, the VAD model can be disabled.
- `vad_kwargs`: Specifies the configurations for the VAD model. `max_single_segment_time`: denotes the maximum duration for audio segmentation by the `vad_model`, with the unit being milliseconds (ms).
- `use_itn`: Whether the output result includes punctuation and inverse text normalization.
- `batch_size_s`: Indicates the use of dynamic batching, where the total duration of audio in the batch is measured in seconds (s).
- `merge_vad`: Whether to merge short audio fragments segmented by the VAD model, with the merged length being `merge_length_s`, in seconds (s).
- `ban_emo_unk`: Whether to ban the output of the `emo_unk` token.
</details>

### Speaker Diarization

This example composes SenseVoiceSmall with separate FSMN-VAD, CAM++, and punctuation models through FunASR. CAM++ provides the speaker labels; the SenseVoiceSmall checkpoint itself does not:

```python
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model = AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    remote_code="./model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    spk_model="cam++",
    punc_model="ct-punc",
    device="cuda:0",
)
res = model.generate(
    input="example.wav",
    cache={},
    language="auto",
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,
    merge_length_s=15,
)
# Per-sentence results with speaker labels
for sent in res[0]["sentence_info"]:
    text = rich_transcription_postprocess(sent["text"])
    print(f"Speaker {sent['spk']}: [{sent['start']}ms - {sent['end']}ms] {text}")
```

> Note: Requires installing FunASR from source: `pip install git+https://github.com/modelscope/FunASR.git`

If all inputs are short audios (<30s), and batch inference is needed to speed up inference efficiency, the VAD model can be removed, and `batch_size` can be set accordingly.
```python
model = AutoModel(model=model_dir, trust_remote_code=True, device="cuda:0")

res = model.generate(
    input=f"{model.model_path}/example/en.mp3",
    cache={},
    language="zh", # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=False,
    batch_size=64, 
)
```

For more usage, please refer to [docs](https://github.com/modelscope/FunASR/blob/main/docs/tutorial/README.md)

### Inference directly

Supports input of audio in any format, with an input duration limit of 30 seconds or less.

```python
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "iic/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device="cuda:0")
m.eval()

res = m.inference(
    data_in=f"{kwargs['model_path']}/example/en.mp3",
    language="auto", # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=False,
    ban_emo_unk=False,
    **kwargs,
)

text = rich_transcription_postprocess(res[0][0]["text"])
print(text)
```

### Export and Test
<details><summary>ONNX and Libtorch Export</summary>

#### ONNX
```python
# pip3 install -U funasr funasr-onnx
from pathlib import Path
from funasr_onnx import SenseVoiceSmall
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess


model_dir = "iic/SenseVoiceSmall"

model = SenseVoiceSmall(model_dir, batch_size=10, quantize=True)

# inference
wav_or_scp = ["{}/.cache/modelscope/hub/{}/example/en.mp3".format(Path.home(), model_dir)]

res = model(wav_or_scp, language="auto", use_itn=True)
print([rich_transcription_postprocess(i) for i in res])
```
Note: ONNX model is exported to the original model directory.

#### Libtorch
```python
from pathlib import Path
from funasr_torch import SenseVoiceSmall
from funasr_torch.utils.postprocess_utils import rich_transcription_postprocess


model_dir = "iic/SenseVoiceSmall"

model = SenseVoiceSmall(model_dir, batch_size=10, device="cuda:0")

wav_or_scp = ["{}/.cache/modelscope/hub/{}/example/en.mp3".format(Path.home(), model_dir)]

res = model(wav_or_scp, language="auto", use_itn=True)
print([rich_transcription_postprocess(i) for i in res])
```
Note: Libtorch model is exported to the original model directory.
</details>

### Run on CPU / edge — llama.cpp / GGUF (no GPU, no Python)

Run SenseVoice as a **single self-contained binary** — this is to SenseVoice what [whisper.cpp](https://github.com/ggml-org/whisper.cpp) is to Whisper, but with far stronger Chinese & Cantonese accuracy. Built-in FSMN-VAD, no Python at runtime.

```bash
bash runtime/llama.cpp/download-funasr-model.sh sensevoice ./gguf
llama-funasr-sensevoice -m ./gguf/sensevoice-small-f16.gguf --vad ./gguf/fsmn-vad.gguf -a audio.wav
```

**Prebuilt binaries:** [Releases](https://github.com/QwenAudio/SenseVoice/releases) · **Download & quickstart:** [funasr.com/llama-cpp](https://www.funasr.com/llama-cpp.html) · **GGUF:** [Hugging Face](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF) · **Docs & benchmarks:** [runtime/llama.cpp/](./runtime/llama.cpp/)

## Service

### Deployment with FastAPI
```shell
export SENSEVOICE_DEVICE=cuda:0
fastapi run --port 50000
```

## Finetune

### Requirements

```shell
git clone https://github.com/modelscope/FunASR.git && cd FunASR
pip3 install -e ./
```
## 🐳 Docker Support

SenseVoice can be built and run using Docker to simplify setup, ensure reproducibility, and support both CPU and GPU inference.

### Build with Docker
```bash
docker build -t sensevoice .
```

### Run (GPU – default)
```bash
docker run --gpus all -p 50000:50000 sensevoice
```
### Run (CPU-only)
```bash
docker run -e SENSEVOICE_DEVICE=cpu -p 50000:50000 sensevoice
```
### Docker Compose
Docker Compose provides an easier way to run SenseVoice with persistent model caching, networking etc. 

### Start Stack
```bash
docker compose up --build
```
### Data prepare

Data examples

```text
{"key": "YOU0000008470_S0000238_punc_itn", "text_language": "<|en|>", "emo_target": "<|NEUTRAL|>", "event_target": "<|Speech|>", "with_or_wo_itn": "<|withitn|>", "target": "Including legal due diligence, subscription agreement, negotiation.", "source": "/cpfs01/shared/Group-speech/beinian.lzr/data/industrial_data/english_all/audio/YOU0000008470_S0000238.wav", "target_len": 7, "source_len": 140}
{"key": "AUD0000001556_S0007580", "text_language": "<|en|>", "emo_target": "<|NEUTRAL|>", "event_target": "<|Speech|>", "with_or_wo_itn": "<|woitn|>", "target": "there is a tendency to identify the self or take interest in what one has got used to", "source": "/cpfs01/shared/Group-speech/beinian.lzr/data/industrial_data/english_all/audio/AUD0000001556_S0007580.wav", "target_len": 18, "source_len": 360}
```

Full ref to `data/train_example.jsonl`

<details><summary>Data Prepare Details</summary>

Description：
- `key`: audio file unique ID
- `source`：path to the audio file
- `source_len`：number of fbank frames of the audio file
- `target`：transcription
- `target_len`：length of target
- `text_language`：language id of the audio file
- `emo_target`：emotion label of the audio file
- `event_target`：event label of the audio file
- `with_or_wo_itn`：whether includes punctuation and inverse text normalization


`train_text.txt`


```bash
BAC009S0764W0121 甚至出现交易几乎停滞的情况
BAC009S0916W0489 湖北一公司以员工名义贷款数十员工负债千万
asr_example_cn_en 所有只要处理 data 不管你是做 machine learning 做 deep learning 做 data analytics 做 data science 也好 scientist 也好通通都要都做的基本功啊那 again 先先对有一些>也许对
ID0012W0014 he tried to think how it could be
```

`train_wav.scp`



```bash
BAC009S0764W0121 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav
BAC009S0916W0489 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0916W0489.wav
asr_example_cn_en https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_cn_en.wav
ID0012W0014 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_en.wav
```

`train_text_language.txt`

The language ids include `<|zh|>`、`<|en|>`、`<|yue|>`、`<|ja|>` and `<|ko|>`.

```bash
BAC009S0764W0121 <|zh|>
BAC009S0916W0489 <|zh|>
asr_example_cn_en <|zh|>
ID0012W0014 <|en|>
```

`train_emo.txt`

The emotion labels include`<|HAPPY|>`、`<|SAD|>`、`<|ANGRY|>`、`<|NEUTRAL|>`、`<|FEARFUL|>`、`<|DISGUSTED|>` and `<|SURPRISED|>`.

```bash
BAC009S0764W0121 <|NEUTRAL|>
BAC009S0916W0489 <|NEUTRAL|>
asr_example_cn_en <|NEUTRAL|>
ID0012W0014 <|NEUTRAL|>
```

`train_event.txt`

The event labels include`<|BGM|>`、`<|Speech|>`、`<|Applause|>`、`<|Laughter|>`、`<|Cry|>`、`<|Sneeze|>`、`<|Breath|>` and `<|Cough|>`.

```bash
BAC009S0764W0121 <|Speech|>
BAC009S0916W0489 <|Speech|>
asr_example_cn_en <|Speech|>
ID0012W0014 <|Speech|>
```

`Command`
```shell
# generate train.jsonl and val.jsonl from wav.scp, text.txt, text_language.txt, emo_target.txt, event_target.txt
sensevoice2jsonl \
++scp_file_list='["../../../data/list/train_wav.scp", "../../../data/list/train_text.txt", "../../../data/list/train_text_language.txt", "../../../data/list/train_emo.txt", "../../../data/list/train_event.txt"]' \
++data_type_list='["source", "target", "text_language", "emo_target", "event_target"]' \
++jsonl_file_out="../../../data/list/train.jsonl"
```

If there is no `train_text_language.txt`, `train_emo_target.txt` and `train_event_target.txt`, the language, emotion and event label will be predicted automatically by using the `SenseVoice` model.
```shell
# generate train.jsonl and val.jsonl from wav.scp and text.txt
sensevoice2jsonl \
++scp_file_list='["../../../data/list/train_wav.scp", "../../../data/list/train_text.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_out="../../../data/list/train.jsonl" \
++model_dir='iic/SenseVoiceSmall'
```
</details>

### Finetune

Ensure to modify the train_tool in finetune.sh to the absolute path of `funasr/bin/train_ds.py` from the FunASR installation directory you have set up earlier.

```shell
bash finetune.sh
```

## WebUI

```shell
python webui.py
```

<div align="center"><img src="image/webui.png" width="700"/> </div>


## Remarkable Third-Party Work
- Triton (GPU) Deployment Best Practices: Using Triton + TensorRT, tested with FP32, achieving an acceleration ratio of 526 on V100 GPU. FP16 support is in progress. [Repository](https://github.com/modelscope/FunASR/blob/main/runtime/triton_gpu/README.md)
- Sherpa-onnx Deployment Best Practices: Supports using SenseVoice in 10 programming languages: C++, C, Python, C#, Go, Swift, Kotlin, Java, JavaScript, and Dart. Also supports deploying SenseVoice on platforms like iOS, Android, and Raspberry Pi. [Repository](https://k2-fsa.github.io/sherpa/onnx/sense-voice/index.html)
- [Orca](https://github.com/stablyai/orca) integrates SenseVoice as local, offline speech-to-text through sherpa-onnx, with automatic Chinese, English, Japanese, Korean, and Cantonese detection on macOS, Linux, and Windows. The integration was [merged in #7436](https://github.com/stablyai/orca/pull/7436) and is available in the [v1.4.159-rc.1 prerelease](https://github.com/stablyai/orca/releases/tag/v1.4.159-rc.1); Orca v1.4.158 stable predates it.
- [SenseVoice.cpp](https://github.com/lovemefan/SenseVoice.cpp). Inference of SenseVoice in pure C/C++ based on GGML, supporting 3-bit, 4-bit, 5-bit, 8-bit quantization, etc. with no third-party dependencies.
- [streaming-sensevoice](https://github.com/pengzhendong/streaming-sensevoice) processes inference in chunks. To achieve pseudo-streaming, it employs a truncated attention mechanism, sacrificing some accuracy. Additionally, this technology supports CTC prefix beam search and hot-word boosting features.
- [OmniSenseVoice](https://github.com/lifeiteng/OmniSenseVoice) is optimized for lightning-fast inference and batching process. 
- [SenseVoice Hotword](https://www.modelscope.cn/models/dengcunqin/SenseVoiceSmall_hotword)，Neural Network Hotword Enhancement，[Contextualized End-to-End Speech Recognition with Contextual Phrase Prediction Network](https://mp.weixin.qq.com/s/1QkIvh8j7rrUjRyWOgAvdA)。
## Ecosystem

SenseVoice is part of the **FunAudioLLM** family:

| Project | Description | Stars |
|---------|-------------|-------|
| [FunASR](https://github.com/modelscope/FunASR) | Industrial speech recognition toolkit — VAD, ASR, punctuation, diarization | [![](https://img.shields.io/github/stars/modelscope/FunASR?style=social)](https://github.com/modelscope/FunASR) |
| [Fun-ASR](https://github.com/QwenAudio/Fun-ASR) | LLM-based ASR family — Nano for zh/en/ja + Chinese dialects; MLT-Nano for 31 languages | [![](https://img.shields.io/github/stars/QwenAudio/Fun-ASR?style=social)](https://github.com/QwenAudio/Fun-ASR) |
| [CosyVoice](https://github.com/QwenAudio/CosyVoice) | Natural speech generation — multi-language, zero-shot cloning | [![](https://img.shields.io/github/stars/QwenAudio/CosyVoice?style=social)](https://github.com/QwenAudio/CosyVoice) |
| [FunClip](https://github.com/modelscope/FunClip) | AI-powered video clipping with speech recognition | [![](https://img.shields.io/github/stars/modelscope/FunClip?style=social)](https://github.com/modelscope/FunClip) |

## License

- Source code in this repository is licensed under the [MIT License](./LICENSE).
- Model weights are distributed separately and follow the terms on each model card. The official [SenseVoiceSmall model card](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) links to the [FunASR Model Open Source License Agreement](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE); other artifacts and conversions may list different terms, so check their model cards before use.
- The maintainers have provided an [official SenseVoiceSmall license clarification](https://github.com/QwenAudio/SenseVoice/issues/334#issuecomment-5083546605) for the [FunASR Model Open Source License Agreement v1.1](https://github.com/modelscope/FunASR/blob/58830eca4012644aac0c3218c3ccc7d98f003fda/MODEL_LICENSE): Commercial use of the official SenseVoiceSmall weights is permitted when the model license is followed; Section 3 is a responsibility and risk disclaimer rather than an additional non-commercial restriction; and fine-tuned derivative weights may remain private. The Section 2.2 attribution and model-name requirements still apply. This clarification covers the official weights only, so check the terms for third-party conversions and bundled artifacts separately.

<a name="Community"></a>
# Community
If you encounter problems in use, you can directly raise Issues on the github page.

You can also scan the following DingTalk group QR code to join the community group for communication and discussion.

|                          FunASR                          |
|:--------------------------------------------------------:|
| <img src="image/dingding_funasr.png" width="250"/></div> |

<a href="https://www.star-history.com/QwenAudio/SenseVoice">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/badge?repo=QwenAudio/SenseVoice&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/badge?repo=QwenAudio/SenseVoice" />
    <img alt="Star History Rank" src="https://api.star-history.com/badge?repo=QwenAudio/SenseVoice" />
  </picture>
</a>
