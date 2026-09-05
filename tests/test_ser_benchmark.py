import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "ser" / "evaluate.py"
)


def load_evaluator():
    spec = importlib.util.spec_from_file_location("ser_evaluate", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_label_accepts_casia_aliases():
    evaluator = load_evaluator()

    assert evaluator.normalize_label("fear") == "fearful"
    assert evaluator.normalize_label("surprise") == "surprised"
    assert evaluator.normalize_label("NEUTRAL") == "neutral"


def test_extract_emotion_reads_raw_sensevoice_tag():
    evaluator = load_evaluator()

    assert (
        evaluator.extract_emotion("<|zh|><|HAPPY|><|Speech|><|withitn|>hello")
        == "happy"
    )


def test_compute_metrics_reports_ua_wa_and_confusion_without_dropping_classes():
    evaluator = load_evaluator()

    result = evaluator.compute_metrics(
        ["angry", "angry", "fearful", "happy"],
        ["angry", "happy", "fearful", "happy"],
    )

    assert result["wa"] == pytest.approx(0.75)
    assert result["ua"] == pytest.approx((0.5 + 1.0 + 1.0) / 3)
    assert result["confusion"]["angry"] == {"angry": 1, "happy": 1}


def test_extract_emotion_rejects_output_without_an_emotion_tag():
    evaluator = load_evaluator()

    with pytest.raises(ValueError, match="emotion tag"):
        evaluator.extract_emotion("plain transcript")


def test_readmes_link_the_ser_reproduction_contract():
    root = Path(__file__).resolve().parents[1]

    for readme in (root / "README.md", root / "README_zh.md"):
        text = readme.read_text(encoding="utf-8")
        assert "benchmarks/ser/README.md" in text
