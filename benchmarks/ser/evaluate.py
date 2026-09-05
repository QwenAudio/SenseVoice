#!/usr/bin/env python3
"""Evaluate zero-shot SenseVoice emotion predictions from a JSONL manifest."""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


CANONICAL_LABELS = (
    "angry",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
)
LABEL_ALIASES = {
    "angry": "angry",
    "fear": "fearful",
    "fearful": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprise": "surprised",
    "surprised": "surprised",
}
EMOTION_TAG = re.compile(r"<\|(?P<label>[A-Z_]+)\|>")


def normalize_label(label):
    """Map dataset and model spellings to the six CASIA evaluation labels."""
    normalized = label.strip().lower()
    try:
        return LABEL_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported emotion label: {label!r}") from exc


def extract_emotion(raw_text):
    """Read SenseVoice's raw emotion control tag before text post-processing."""
    for match in EMOTION_TAG.finditer(raw_text):
        label = match.group("label").lower()
        if label in LABEL_ALIASES:
            return normalize_label(label)
    raise ValueError(f"SenseVoice output has no supported emotion tag: {raw_text!r}")


def compute_metrics(references, predictions):
    """Return weighted accuracy, unweighted accuracy, and a sparse confusion map."""
    if not references:
        raise ValueError("no evaluation records")
    if len(references) != len(predictions):
        raise ValueError("reference and prediction counts differ")

    confusion = defaultdict(Counter)
    totals = Counter()
    correct = Counter()
    for reference, prediction in zip(references, predictions):
        reference = normalize_label(reference)
        prediction = normalize_label(prediction)
        totals[reference] += 1
        confusion[reference][prediction] += 1
        if reference == prediction:
            correct[reference] += 1

    recalls = {label: correct[label] / totals[label] for label in totals}
    return {
        "records": len(references),
        "wa": sum(correct.values()) / len(references),
        "ua": sum(recalls.values()) / len(recalls),
        "recall_by_label": recalls,
        "confusion": {label: dict(confusion[label]) for label in sorted(confusion)},
    }


def read_manifest(path):
    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record.get("audio"), str) or not isinstance(record.get("label"), str):
            raise ValueError(f"manifest line {line_number} needs string audio and label fields")
        records.append(record)
    if not records:
        raise ValueError("manifest contains no records")
    return records


def evaluate(manifest, model_name, device):
    from funasr import AutoModel

    model = AutoModel(model=model_name, device=device, disable_update=True)
    references, predictions = [], []
    for record in read_manifest(manifest):
        result = model.generate(input=record["audio"], language="auto", use_itn=True)
        raw_text = result[0]["text"]
        references.append(record["label"])
        predictions.append(extract_emotion(raw_text))
    return compute_metrics(references, predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="JSONL records with audio and label fields")
    parser.add_argument("--model", default="iic/SenseVoiceSmall")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = evaluate(args.manifest, args.model, args.device)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
