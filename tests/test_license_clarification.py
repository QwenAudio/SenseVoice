from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_CLARIFICATION = (
    "https://github.com/QwenAudio/SenseVoice/issues/334#issuecomment-5083546605"
)
IMMUTABLE_MODEL_LICENSE = (
    "https://github.com/modelscope/FunASR/blob/"
    "58830eca4012644aac0c3218c3ccc7d98f003fda/MODEL_LICENSE"
)


def test_readmes_surface_official_sensevoicesmall_commercial_use_clarification():
    expected_markers = {
        "README.md": [
            "Agreement v1.1",
            "Commercial use of the official SenseVoiceSmall weights is permitted",
            "when the model license is followed",
            "Section 3 is a responsibility and risk disclaimer",
            "fine-tuned derivative weights may remain private",
            "Section 2.2 attribution and model-name requirements",
            "official weights only",
            "third-party conversions and bundled artifacts separately",
        ],
        "README_zh.md": [
            "模型开源协议 v1.1",
            "允许商业使用官方 SenseVoiceSmall 权重",
            "遵守模型协议时",
            "第 3 节属于责任和风险免责声明",
            "微调后的衍生权重可以保持私有",
            "第 2.2 节的署名和模型名称要求",
            "仅适用于官方权重",
            "第三方转换版本和打包制品仍需分别核对",
        ],
    }

    for relpath, markers in expected_markers.items():
        text = (ROOT / relpath).read_text()
        assert OFFICIAL_CLARIFICATION in text, (
            f"{relpath} is missing the official license clarification"
        )
        assert IMMUTABLE_MODEL_LICENSE in text, (
            f"{relpath} is missing the immutable v1.1 model license"
        )
        for marker in markers:
            assert marker in text, f"{relpath} is missing: {marker}"


def test_license_clarification_keeps_code_and_weight_terms_distinct():
    for relpath in ["README.md", "README_zh.md"]:
        text = (ROOT / relpath).read_text()
        assert "https://huggingface.co/FunAudioLLM/SenseVoiceSmall" in text
        assert "https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE" in text

    english = (ROOT / "README.md").read_text()
    chinese = (ROOT / "README_zh.md").read_text()
    assert "Source code in this repository is licensed under the" in english
    assert "official SenseVoiceSmall weights are licensed under the MIT" not in english
    assert "本仓库源码采用" in chinese
    assert "官方 SenseVoiceSmall 权重采用 MIT" not in chinese
