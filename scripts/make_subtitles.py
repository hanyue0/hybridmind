"""把配音转成带绝对时间轴的字幕（SRT）。

思路：7 段录音是分开录的，逐段转写再整体偏移到视频时间轴上，
比直接对整条音轨做识别更准（段落边界已知，不会漂移）。

用法：
    python scripts/make_subtitles.py
输出：
    docs/hybridmind.srt
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.assemble_final import TIMELINE, VOICE  # noqa: E402

SRT = ROOT / "docs" / "hybridmind.srt"
MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
MAX_CHARS = 24  # 单行中文字数上限


def fmt(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def split_long(text: str) -> list[str]:
    """按标点把过长的句子切成不超过 MAX_CHARS 的短句。"""
    text = text.strip()
    if len(text) <= MAX_CHARS:
        return [text] if text else []
    parts = re.split(r"(?<=[，。！？；、])", text)
    out: list[str] = []
    buf = ""
    for part in parts:
        if len(buf) + len(part) <= MAX_CHARS:
            buf += part
        else:
            if buf:
                out.append(buf)
            buf = part if len(part) <= MAX_CHARS else part[:MAX_CHARS]
            rest = part[MAX_CHARS:]
            while len(rest) > MAX_CHARS:
                out.append(rest[:MAX_CHARS])
                rest = rest[MAX_CHARS:]
            buf = rest
    if buf:
        out.append(buf)
    return [line for line in out if line.strip()]


def main() -> None:
    from faster_whisper import WhisperModel

    print(f"加载模型 {MODEL_SIZE}（首次会下载权重）…")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    entries: list[tuple[float, float, str]] = []
    cursor = 0.0
    for _, duration, voice, offset in TIMELINE:
        if voice:
            path = VOICE / voice
            print(f"转写 {voice} …")
            segments, _ = model.transcribe(
                str(path), language="zh", vad_filter=True, beam_size=5
            )
            base = cursor + offset
            for seg in segments:
                for piece in split_long(seg.text):
                    # 句内再按字数等比分配时间
                    share = len(piece) / max(len(seg.text.strip()), 1)
                    start = base + seg.start + (seg.end - seg.start) * (1 - share) * 0
                    entries.append((base + seg.start, base + seg.end, piece))
        cursor += duration

    # 去重与合并：同一秒内的碎句合并
    merged: list[tuple[float, float, str]] = []
    for start, end, text in entries:
        if merged and start - merged[-1][1] < 0.15 and len(merged[-1][2]) + len(text) <= MAX_CHARS * 2:
            prev = merged.pop()
            merged.append((prev[0], max(prev[1], end), prev[2] + text))
        else:
            merged.append((start, end, text))

    lines = []
    for index, (start, end, text) in enumerate(merged, 1):
        lines.append(f"{index}\n{fmt(start)} --> {fmt(max(end, start + 0.6))}\n{text}\n")
    SRT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n字幕：{SRT}")
    print(f"条数：{len(merged)}")


if __name__ == "__main__":
    main()
