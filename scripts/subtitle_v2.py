"""给 v2 视频做字幕：whisper 逐词时间戳 + 逐字稿强制对齐 → SRT → 烧进视频。

为什么不能按字数比例分时间：语速与停顿并不均匀，比例分配在长段落会漂移
（上一版实测 2:10 之后字幕跑在语音前面）。这里的做法是：

1. 用 faster-whisper 转写每段配音，拿**逐词**起止时间；
2. 把识别文本与逐字稿按字符做 difflib 对齐，把逐字稿的每个字映射到真实时间；
3. 按标点切字幕行，取首字开始 / 末字结束；
4. 加上该段在成片里的起始偏移，输出全局 SRT。

识别文本与逐字稿有出入没关系（如「数据集」被识别成「数据级」），
对齐算法允许替换与插入，只要顺序一致即可。

用法：
    python scripts/subtitle_v2.py                    # 生成 SRT
    python scripts/subtitle_v2.py --burn             # 生成并烧进视频
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VIDEO = ROOT / "docs" / "video" / "v2"
CACHE = VIDEO / "asr_words.json"
GAP = 0.6
NAMES = ["seg1", "seg2", "seg3", "seg4", "seg5", "seg6"]


def ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def duration(path: Path) -> float:
    r = subprocess.run([ffmpeg(), "-i", str(path)], capture_output=True)
    for line in r.stderr.decode("utf-8", "ignore").splitlines():
        if "Duration:" in line:
            h, m, s = line.split("Duration:")[1].split(",")[0].strip().split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    raise SystemExit(f"读不到时长：{path}")


def script_segments(audio_dir: Path) -> dict:
    """逐字稿：优先用配音脚本里写死的文本（权威），而不是识别结果。"""
    from scripts.make_voiceover import SEGMENTS

    return {name: text for name, text in SEGMENTS}


def transcribe(audio_dir: Path) -> dict:
    """逐段缓存：只转写缺失或时长相符不上的段，改一段不用重跑全部。"""
    cached = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    missing = []
    for name in NAMES:
        seg_dur = duration(audio_dir / f"{name}.mp3")
        words = cached.get(name) or []
        last_end = max((w["e"] for w in words), default=0.0)
        if not words or abs(last_end - seg_dur) > 2.0:
            missing.append(name)
    if not missing:
        print(f"复用转写缓存 {CACHE.name}")
        return cached

    from faster_whisper import WhisperModel

    print(f"加载 whisper small（待转写：{', '.join(missing)}）…")
    model = WhisperModel("small", device="cpu", compute_type="int8")
    for name in missing:
        src = audio_dir / f"{name}.mp3"
        segments, _ = model.transcribe(
            str(src), language="zh", vad_filter=True, beam_size=5, word_timestamps=True
        )
        words = []
        for seg in segments:
            for w in seg.words or []:
                words.append({"w": w.word, "s": float(w.start), "e": float(w.end)})
        cached[name] = words
        print(f"  {name}: {len(words)} 词")
        CACHE.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
    return cached


def char_times(words: list[dict]):
    chars, starts, ends = [], [], []
    for word in words:
        text = word["w"]
        n = max(len(text), 1)
        for i, ch in enumerate(text):
            chars.append(ch)
            starts.append(word["s"] + (word["e"] - word["s"]) * i / n)
            ends.append(word["s"] + (word["e"] - word["s"]) * (i + 1) / n)
    return "".join(chars), starts, ends


def align(script: str, asr: str, starts: list[float], ends: list[float]):
    matcher = difflib.SequenceMatcher(None, asr, script, autojunk=False)
    result = [(0.0, 0.0)] * len(script)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(j2 - j1):
                result[j1 + k] = (starts[i1 + k], ends[i1 + k])
        elif tag == "replace" and (i2 - i1) > 0:
            span_start, span_end = starts[i1], ends[min(i2, len(ends)) - 1]
            n = max(j2 - j1, 1)
            for k in range(j2 - j1):
                result[j1 + k] = (
                    span_start + (span_end - span_start) * k / n,
                    span_start + (span_end - span_start) * (k + 1) / n,
                )
        else:
            left = result[j1 - 1][1] if j1 > 0 else (starts[0] if starts else 0.0)
            right = starts[min(i1, len(starts) - 1)] if starts else left + 1
            n = max(j2 - j1, 1)
            for k in range(j2 - j1):
                result[j1 + k] = (
                    left + (right - left) * k / n,
                    left + (right - left) * (k + 1) / n,
                )
    return result


PUNCT = "。！？；，"


def split_cues(text: str, max_len: int = 22) -> list[str]:
    """按标点切句，再贪心合并到不超过 max_len，避免出现「？」单独一屏。"""
    tokens, buf = [], ""
    for ch in text:
        buf += ch
        if ch in "。！？；：，":
            tokens.append(buf)
            buf = ""
    if buf.strip():
        tokens.append(buf)

    cues: list[str] = []
    for token in tokens:
        while len(token) > max_len:
            # 超长句按长度硬切，但尾部至少留 4 个字，避免出现孤零零的标点
            cut = max_len if len(token) - max_len >= 4 else len(token) - 4
            cues.append(token[:cut])
            token = token[cut:]
        if token:
            if cues and len(cues[-1]) + len(token) <= max_len:
                cues[-1] += token
            else:
                cues.append(token)
    return [c.strip() for c in cues if c.strip()]


def fmt(t: float) -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def build_srt(audio_dir: Path) -> Path:
    scripts = script_segments(audio_dir)
    words_by = transcribe(audio_dir)

    entries = []
    cursor = 0.0
    for name in NAMES:
        seg_len = duration(audio_dir / f"{name}.mp3")
        script = scripts[name]
        words = words_by.get(name, [])
        if words:
            asr_text, starts, ends = char_times(words)
            times = align(script, asr_text, starts, ends)
        else:
            times = [(0.0, seg_len)] * len(script)

        pos = 0
        for cue in split_cues(script):
            idx = script.find(cue, pos)
            if idx < 0:
                idx = pos
            pos = idx + len(cue)
            start = times[idx][0] if idx < len(times) else 0.0
            last = min(idx + len(cue) - 1, len(times) - 1)
            end = times[last][1] if last >= 0 else start + 1.2
            start = min(max(0.0, start), seg_len)
            end = min(max(end, start + 0.7), seg_len)
            entries.append((cursor + start, cursor + end, cue))
        cursor += seg_len + GAP

    out = VIDEO / "HybridMind_v2.srt"
    blocks = []
    for i, (s, e, text) in enumerate(entries, 1):
        blocks.append(f"{i}\n{fmt(s)} --> {fmt(e)}\n{text}\n")
    out.write_text("\n".join(blocks), encoding="utf-8")
    print(f"字幕 {len(entries)} 条 -> {out}（最后一屏结束于 {fmt(entries[-1][1])}）")
    return out


def burn(video: Path, srt: Path, out: Path) -> None:
    style = (
        "FontName=Microsoft YaHei,FontSize=17,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H80000000,BorderStyle=4,Outline=0,Shadow=1.2,BackColour=&H99000000,"
        "Alignment=2,MarginV=30"
    )
    # Windows 上不能把 "D:/..." 直接塞进 subtitles= 滤镜（冒号会被当成参数分隔），
    # 所以切到 SRT 所在目录、只传文件名
    cmd = [
        ffmpeg(), "-y", "-i", str(video.resolve()),
        "-vf", f"subtitles={srt.name}:force_style='{style}'",
        "-c:v", "libx264", "-crf", "20", "-preset", "medium",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out.resolve()),
    ]
    r = subprocess.run(cmd, cwd=str(srt.parent), capture_output=True)
    if r.returncode != 0:
        print(r.stderr.decode("utf-8", "ignore")[-1500:])
        raise SystemExit("烧字幕失败")
    print(f"已输出 {out}（{out.stat().st_size // 1024 // 1024} MB）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=str(ROOT / "data" / "voiceover_v2"))
    ap.add_argument("--video", default=str(VIDEO / "HybridMind_演示视频_v2.mp4"))
    ap.add_argument("--out", default=str(VIDEO / "HybridMind_演示视频_v2_字幕版.mp4"))
    ap.add_argument("--burn", action="store_true")
    args = ap.parse_args()

    srt = build_srt(Path(args.audio))
    if args.burn:
        burn(Path(args.video), srt, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

