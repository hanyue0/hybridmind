"""用逐词时间戳把逐字稿和语音对齐，生成严格同步的字幕。

上一版的做法是「按字数比例把整段文字铺进 ASR 的句子时间窗」，
在语速不均的段落会明显漂移（实测 2:10 之后字幕跑在语音前面）。

这一版改为：
1. 取 whisper 的 word_timestamps，拿到每个词的起止时刻；
2. 用 difflib 把「识别文本」和「逐字稿」按字符对齐，
   从而把逐字稿的每个字映射到真实时间；
3. 再按标点切成字幕行，取该行首字的开始时刻与末字的结束时刻。

识别文本与逐字稿不完全一致也没关系——对齐算法允许插入/替换，
只要整体顺序对得上即可。

用法：
    python scripts/align_subtitles.py
输出：
    docs/asr_words.json（缓存，避免重复转写）
    docs/hybridmind.srt（覆盖）
"""

from __future__ import annotations

import difflib
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.assemble_final import TIMELINE, VOICE  # noqa: E402
from scripts.refine_subtitles import fmt, script_narrations, split_lines  # noqa: E402

SRT = ROOT / "docs" / "hybridmind.srt"
CACHE = ROOT / "docs" / "asr_words.json"
MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")


def transcribe_all() -> dict:
    if CACHE.exists():
        print(f"复用缓存 {CACHE}")
        return json.loads(CACHE.read_text(encoding="utf-8"))

    from faster_whisper import WhisperModel

    print(f"加载 {MODEL_SIZE} …")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    data: dict[str, list[dict]] = {}
    for _, _, voice, _ in TIMELINE:
        if not voice:
            continue
        print(f"转写 {voice} …")
        segments, _ = model.transcribe(
            str(VOICE / voice), language="zh", vad_filter=True,
            beam_size=5, word_timestamps=True,
        )
        words = []
        for seg in segments:
            for w in seg.words or []:
                words.append({"w": w.word, "s": float(w.start), "e": float(w.end)})
        data[voice] = words
    CACHE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"已缓存 {CACHE}")
    return data


def char_times(words: list[dict]) -> tuple[str, list[float], list[float]]:
    """把词序列摊平成字符序列，同时给出每个字符的起止时间。"""
    chars, starts, ends = [], [], []
    for word in words:
        text = word["w"]
        n = max(len(text), 1)
        for i, ch in enumerate(text):
            chars.append(ch)
            starts.append(word["s"] + (word["e"] - word["s"]) * i / n)
            ends.append(word["s"] + (word["e"] - word["s"]) * (i + 1) / n)
    return "".join(chars), starts, ends


def align(script: str, asr: str, starts: list[float], ends: list[float]) -> list[tuple[float, float]]:
    """把逐字稿的每个字符映射到时间，返回与 script 等长的 (start, end) 列表。"""
    matcher = difflib.SequenceMatcher(None, asr, script, autojunk=False)
    result: list[tuple[float, float]] = [(0.0, 0.0)] * len(script)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(j2 - j1):
                result[j1 + k] = (starts[i1 + k], ends[i1 + k])
        elif tag == "replace" and (i2 - i1) > 0:
            # 识别与原文在此处有差异：整段按比例摊到对应的识别字符上
            span_start, span_end = starts[i1], ends[min(i2, len(ends)) - 1]
            n = max(j2 - j1, 1)
            for k in range(j2 - j1):
                result[j1 + k] = (span_start + (span_end - span_start) * k / n,
                                  span_start + (span_end - span_start) * (k + 1) / n)
        else:
            # insert（逐字稿多出内容）或 delete：用相邻已知时间插值
            left = result[j1 - 1][1] if j1 > 0 else (starts[0] if starts else 0.0)
            right = starts[min(i1, len(starts) - 1)] if starts else left + 1
            n = max(j2 - j1, 1)
            for k in range(j2 - j1):
                result[j1 + k] = (left + (right - left) * k / n, left + (right - left) * (k + 1) / n)
    return result


def main() -> None:
    words_by_file = transcribe_all()
    narrations = script_narrations()

    entries: list[tuple[float, float, str]] = []
    cursor = 0.0
    narration_index = 0
    for _, duration, voice, offset in TIMELINE:
        if not voice:
            cursor += duration
            continue
        base = cursor + offset
        script = narrations[narration_index] if narration_index < len(narrations) else ""
        narration_index += 1
        words = words_by_file.get(voice, [])
        if not words or not script:
            cursor += duration
            continue

        asr_text, starts, ends = char_times(words)
        times = align(script, asr_text, starts, ends)

        # 按标点切行，行内保持连续
        pos = 0
        for line in split_lines(script):
            idx = script.find(line, pos)
            if idx < 0:
                idx = pos
            line_start = times[idx][0] if idx < len(times) else 0.0
            last = min(idx + len(line) - 1, len(times) - 1)
            line_end = times[last][1] if last >= 0 else line_start + 1.5
            entries.append((base + line_start, base + max(line_end, line_start + 0.5), line))
            pos = idx + len(line)
        cursor += duration

    lines = []
    for index, (start, end, text) in enumerate(entries, 1):
        lines.append(f"{index}\n{fmt(start)} --> {fmt(end)}\n{text}\n")
    SRT.write_text("\n".join(lines), encoding="utf-8")
    print(f"对齐完成：{len(entries)} 条 -> {SRT}")


if __name__ == "__main__":
    main()
