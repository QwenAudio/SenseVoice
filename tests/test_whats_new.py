from pathlib import Path


def whats_new_bullets(path, heading):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    start = lines.index(heading) + 1
    section = []
    for line in lines[start:]:
        if line.startswith("# "):
            break
        section.append(line)
    return [line for line in section if line.startswith("- ")]


def test_whats_new_is_a_current_three_item_summary():
    root = Path(__file__).resolve().parents[1]
    sections = (
        (root / "README.md", "# What's New 🔥"),
        (root / "README_zh.md", "# 最新动态 🔥"),
        (root / "README_ja.md", "# 最新情報 🔥"),
    )

    for path, heading in sections:
        text = path.read_text(encoding="utf-8")
        bullets = whats_new_bullets(path, heading)
        assert len(bullets) == 3
        assert 'funasr==1.4.14' in text
        assert "funasr==1.3.29" not in text
        assert "https://github.com/QwenAudio/SenseVoice/releases" in text
