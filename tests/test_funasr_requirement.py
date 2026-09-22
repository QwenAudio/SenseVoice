import re
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]


def test_funasr_minimum_version_matches_current_examples():
    requirements = (ROOT / "requirements.txt").read_text()
    assert "funasr>=1.3.26" in requirements
    assert "funasr>=1.1.3" not in requirements
    assert "funasr>=1.3.23" not in requirements


def test_readmes_explain_upgrade_for_existing_installs():
    for readme in ["README.md", "README_zh.md", "README_ja.md"]:
        text = (ROOT / readme).read_text()
        assert 'pip install -U "funasr>=1.3.26"' in text
        assert "funasr>=1.3.23" not in text


def test_readmes_surface_sensevoice_gguf_edge_path():
    required_links = [
        "https://www.funasr.com/llama-cpp.html",
        "https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF",
    ]
    for readme in ["README.md", "README_zh.md", "README_ja.md"]:
        text = (ROOT / readme).read_text()
        for link in required_links:
            assert link in text


def test_readmes_surface_current_funasr_release():
    required = [
        "funasr==1.4.14",
        "https://github.com/modelscope/FunASR/releases/tag/v1.4.14",
        "https://github.com/QwenAudio/SenseVoice/releases",
    ]
    for relpath in ["README.md", "README_zh.md", "README_ja.md"]:
        text = (ROOT / relpath).read_text()
        for marker in required:
            assert marker in text, f"{relpath} is missing {marker}"


def test_readmes_surface_orca_sensevoice_desktop_integration():
    required_links = [
        "https://github.com/stablyai/orca",
        "https://github.com/stablyai/orca/pull/7436",
        "https://github.com/stablyai/orca/releases/tag/v1.4.206",
    ]
    release_markers = {
        "README.md": ("stable release", "Download the SenseVoice model", "non-streaming"),
        "README_zh.md": ("稳定版", "先下载 SenseVoice 模型", "非流式"),
        "README_ja.md": ("安定版", "SenseVoice モデルをダウンロード", "非ストリーミング"),
    }
    for relpath, markers in release_markers.items():
        text = (ROOT / relpath).read_text()
        orca_lines = [line for line in text.splitlines() if line.startswith("- [Orca]")]
        assert len(orca_lines) == 1, f"{relpath} must have one Orca entry"
        entry = orca_lines[0]
        for marker in ["Orca", "SenseVoice", *required_links, *markers]:
            assert marker in entry, f"{relpath} is missing {marker} in its Orca entry"
        assert "v1.4.159-rc.1" not in entry
        assert "v1.4.158" not in entry


def test_readme_relative_markdown_links_point_to_existing_files():
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for relpath in ["README.md", "README_zh.md"]:
        readme_path = ROOT / relpath
        for target in link_pattern.findall(readme_path.read_text()):
            parsed = urlparse(target)
            if parsed.scheme or parsed.netloc or target.startswith("#"):
                continue
            link_path = unquote(parsed.path)
            if not link_path or link_path.startswith("#"):
                continue
            resolved = (readme_path.parent / link_path).resolve()
            assert resolved.exists(), f"{relpath} links to missing file: {target}"
