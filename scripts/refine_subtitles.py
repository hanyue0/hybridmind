"""用逐字稿替换 ASR 识别文本，只保留其时间戳。

为什么这么做：whisper 的中文识别有错字（"一分钱并花"），而且切句过长。
我们手里有作者写的逐字稿，是权威文本；ASR 提供的只是"这句话在什么时候说完"。
于是按每段的时间窗口，把逐字稿按字数比例铺进去，既准确又可读。

用法：
    python scripts/refine_subtitles.py
输出：
    docs/hybridmind.srt（覆盖）
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.assemble_final import TIMELINE  # noqa: E402

SRT = ROOT / "docs" / "hybridmind.srt"
SCRIPT = ROOT / "docs" / "tech" / "video_script.md"
MAX_CHARS = 22


def fmt(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt(path: Path) -> list[tuple[float, float]]:
    times: list[tuple[float, float]] = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")):
        m = re.search(r"(\d+):(\d+):(\d+),(\d+)\s*-->\s*(\d+):(\d+):(\d+),(\d+)", block)
        if m:
            a = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000
            b = int(m[5]) * 3600 + int(m[6]) * 60 + int(m[7]) + int(m[8]) / 1000
            times.append((a, b))
    return times


def script_narrations() -> list[str]:
    """从拍摄脚本里取出 7 段口播文字。

    只在「**口播**」标记之后收集引用行——段 4 前面还有一行"按画面节奏念"
    的数字提示（> 0.97 / 95 / …），如果不加限定会被当成正文，
    导致该段之后全部字幕错位。
    """
    text = SCRIPT.read_text(encoding="utf-8")
    blocks = re.split(r"^### 段 ", text, flags=re.M)[1:]
    out: list[str] = []
    for block in blocks:
        lines: list[str] = []
        collecting = False
        for raw in block.splitlines():
            line = raw.strip()
            if "口播" in line:
                collecting = True
                continue
            if not collecting:
                continue
            if not line:
                continue          # 口播标记与正文之间有空行
            if not line.startswith(">"):
                break
            cleaned = line.lstrip(">").strip()
            if cleaned:
                lines.append(cleaned)
        out.append("".join(lines))
    return out


def split_lines(text: str, limit: int = MAX_CHARS) -> list[str]:
    parts = re.split(r"(?<=[，。！？；、])", text)
    out, buf = [], ""
    for part in parts:
        if not part:
            continue
        if len(buf) + len(part) <= limit:
            buf += part
        else:
            if buf:
                out.append(buf)
            while len(part) > limit:
                out.append(part[:limit])
                part = part[limit:]
            buf = part
    if buf:
        out.append(buf)
    # 去掉切句产生的行首标点与空白
    cleaned = [line.strip("，。、；： ").strip() for line in out]
    return [line for line in cleaned if line]


def main() -> None:
    windows = parse_srt(SRT)
    narrations = script_narrations()
    if len(narrations) != 7:
        print(f"警告：解析到 {len(narrations)} 段口播（预期 7）")

    # 每段在视频中的绝对起点
    starts, cursor = [], 0.0
    for _, duration, voice, offset in TIMELINE:
        starts.append(cursor + offset if voice else cursor)
        cursor += duration

    narration_index = 0
    entries: list[tuple[float, float, str]] = []
    cursor = 0.0
    for _, duration, voice, offset in TIMELINE:
        if not voice:
            cursor += duration
            continue
        seg_start = cursor + offset
        seg_end = cursor + duration
        bucket = [w for w in windows if seg_start - 0.2 <= w[0] < seg_end]
        text = narrations[narration_index] if narration_index < len(narrations) else ""
        narration_index += 1
        if not bucket or not text:
            cursor += duration
            continue

        total_span = sum(max(b - a, 0.4) for a, b in bucket)
        pos = 0
        for a, b in bucket:
            share = max(b - a, 0.4) / total_span
            take = max(1, round(len(text) * share))
            chunk = text[pos:pos + take]
            pos += take
            if not chunk.strip():
                continue
            pieces = split_lines(chunk)
            sub_span = max(b - a, 0.4)
            unit = sub_span / max(len(chunk), 1)
            offset_chars = 0
            for piece in pieces:
                p_start = a + unit * offset_chars
                p_end = p_start + unit * len(piece)
                entries.append((p_start, min(p_end, seg_end), piece))
                offset_chars += len(piece)
        # 剩余的尾部文字并入最后一条
        if pos < len(text):
            tail = text[pos:]
            last_a, last_b = bucket[-1]
            entries.append((last_b, min(last_b + 2.5, seg_end), tail[:MAX_CHARS]))
        cursor += duration

    lines = []
    for index, (start, end, text) in enumerate(entries, 1):
        lines.append(f"{index}\n{fmt(start)} --> {fmt(max(end, start + 0.8))}\n{text}\n")
    SRT.write_text("\n".join(lines), encoding="utf-8")
    print(f"重写字幕：{len(entries)} 条 -> {SRT}")


if __name__ == "__main__":
    main()
