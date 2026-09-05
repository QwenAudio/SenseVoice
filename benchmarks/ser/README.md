# Reproducing SenseVoice SER Measurements

The SER table in the project README reports zero-shot results. It is not a
promise that every dataset mirror, parser, or aggregate metric produces the
same number. In particular, the CASIA SenseVoiceSmall row is **70.0 UA / 70.0
WA** for the six-label benchmark protocol shown in the table.

Use `evaluate.py` to make a local evaluation reproducible. It accepts a JSONL
manifest; every non-empty line must contain an audio path or URL and one of the
following labels:

```json
{"audio": "/data/CASIA/angry/example.wav", "label": "angry"}
```

The accepted canonical labels are `angry`, `fearful`, `happy`, `neutral`,
`sad`, and `surprised`. Dataset spellings `fear` and `surprise` are normalized
to `fearful` and `surprised`.

```bash
python benchmarks/ser/evaluate.py casia.jsonl \
  --model iic/SenseVoiceSmall --device cuda:0 --output casia-results.json
```

The evaluator reads the raw `<|EMOTION|>` tag returned by SenseVoice before
calling rich-text post-processing. Do not infer the label by splitting the
formatted transcription: tags and display text have different contracts.

`wa` is accuracy over all records. `ua` is the mean recall over labels present
in the manifest. The JSON result includes per-label recall and a confusion map;
unknown labels, missing emotion tags, and malformed manifest records fail the
run instead of being skipped.

CASIA and RAVDESS distributions are controlled by their respective providers.
Keep the dataset version, manifest, model revision, package versions, and this
JSON result together when comparing a rerun with the README table.
