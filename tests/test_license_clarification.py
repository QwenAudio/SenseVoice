from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION_FOLLOWUP = (
    "https://github.com/QwenAudio/SenseVoice/issues/334#issuecomment-5463240506"
)
IMMUTABLE_MODEL_LICENSE = (
    "https://github.com/modelscope/FunASR/blob/"
    "58830eca4012644aac0c3218c3ccc7d98f003fda/MODEL_LICENSE"
)


def test_readmes_surface_pending_license_owner_confirmation():
    expected_markers = {
        "README.md": [
            "Agreement v1.1",
            "remain open",
            "license-owner/core-maintainer confirmation",
            "not be treated as final confirmation",
        ],
        "README_zh.md": [
            "模型开源协议 v1.1",
            "仍未关闭",
            "许可方/核心维护者确认",
            "不应将早期回复视为最终确认",
        ],
    }

    for relpath, markers in expected_markers.items():
        text = (ROOT / relpath).read_text()
        assert CONFIRMATION_FOLLOWUP in text, (
            f"{relpath} is missing the pending-confirmation follow-up"
        )
        assert IMMUTABLE_MODEL_LICENSE in text, (
            f"{relpath} is missing the immutable v1.1 model license"
        )
        for marker in markers:
            assert marker in text, f"{relpath} is missing: {marker}"


def test_readmes_do_not_present_the_earlier_reply_as_final_authorization():
    english = (ROOT / "README.md").read_text()
    chinese = (ROOT / "README_zh.md").read_text()
    assert "official SenseVoiceSmall license clarification" not in english
    assert "Commercial use of the official SenseVoiceSmall weights is permitted" not in english
    assert "fine-tuned derivative weights may remain private" not in english
    assert "SenseVoiceSmall 官方许可澄清" not in chinese
    assert "允许商业使用官方 SenseVoiceSmall 权重" not in chinese
    assert "微调后的衍生权重可以保持私有" not in chinese


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
