"""按实际配音时长重新排布时间轴，合成最终成片。

为什么要重排：原视频按脚本预估的 6:30 剪辑，而实际录音总长 4:19。
直接贴上去会有大量死气。这里以「每段视频时长 ≈ 该段配音 + 呼吸间隔」重算，
最终 5:02，仍满足 5–8 分钟的要求。

用法：
    python scripts/assemble_final.py
输出：
    docs/HybridMind_演示视频_成片.mp4
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "docs" / "video"
FRAMES = VIDEO / "frames"
CLIPS = VIDEO / "clips_final"
VOICE = Path(r"D:\桌面\AMC-传智\配音")
OUT = ROOT / "docs" / "HybridMind_演示视频_成片.mp4"
LIVE = VIDEO / "live_demo.mp4"

# (卡片名 或 __live__, 时长秒, 配音文件 或 None, 配音相对本段的起始偏移)
TIMELINE = [
    ("s0_title", 8, None, 0.0),
    ("s1_contrast", 35, "段1-痛点.mp3", 1.5),
    ("s2_reversal", 36, "段2-打脸.mp3", 1.5),
    ("s3_design", 47, "段3-方案.mp3", 1.5),
    ("__live__", 59, "段4-现场演示.mp3", 3.0),
    ("s5_ood", 32, "段5-数据.mp3", 1.5),
    ("s5_curve", 31, None, 0.0),      # 段5 的配音从上一张卡片开始，贯穿两张
    ("s6_engineer", 28, "段6-工程.mp3", 1.5),
    ("s7_close", 26, "段7-收尾.mp3", 1.5),
]


def ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def build_clips(ff: str) -> list[Path]:
    CLIPS.mkdir(parents=True, exist_ok=True)
    pieces: list[Path] = []
    for index, (name, duration, _, _) in enumerate(TIMELINE):
        if name == "__live__":
            pieces.append(LIVE)
            continue
        png = FRAMES / f"{name}.png"
        if not png.exists():
            raise SystemExit(f"缺少卡片：{png}")
        clip = CLIPS / f"{index:02d}_{name}.mp4"
        clip.unlink(missing_ok=True)
        subprocess.run(
            [ff, "-hide_banner", "-loglevel", "error",
             "-loop", "1", "-i", str(png), "-t", str(duration),
             "-r", "30", "-vf", "scale=1920:1080,setsar=1",
             "-c:v", "libx264", "-preset", "medium", "-crf", "20",
             "-pix_fmt", "yuv420p", "-y", str(clip)],
            capture_output=True, timeout=300, check=True,
        )
        pieces.append(clip)
        print(f"片段 {name}: {duration}s")
    return pieces


def main() -> None:
    ff = ffmpeg()
    pieces = build_clips(ff)

    listfile = CLIPS / "list.txt"
    listfile.write_text("\n".join(f"file '{p.resolve().as_posix()}'" for p in pieces) + "\n", encoding="utf-8")

    total = sum(d for _, d, _, _ in TIMELINE)

    # 计算每段配音的绝对起点
    starts = []
    cursor = 0.0
    for name, duration, voice, offset in TIMELINE:
        if voice:
            starts.append((voice, cursor + offset))
        cursor += duration

    # 输入：0 = 视频，其余 = 各段配音
    inputs = ["-f", "concat", "-safe", "0", "-i", str(listfile)]
    for voice, _ in starts:
        inputs += ["-i", str(VOICE / voice)]

    # 配音依次延迟到各自起点后混合
    filters = []
    labels = []
    for i, (voice, start) in enumerate(starts, start=1):
        ms = int(round(start * 1000))
        filters.append(f"[{i}:a]adelay={ms}|{ms},aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a{i}]")
        labels.append(f"[a{i}]")
    filters.append("".join(labels) + f"amix=inputs={len(labels)}:normalize=0:dropout_transition=0[mix]")
    filters.append("[mix]loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[aout]")

    cmd = [ff, "-hide_banner", "-loglevel", "error", *inputs,
           "-filter_complex", ";".join(filters),
           "-map", "0:v", "-map", "[aout]",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-t", str(total), "-movflags", "+faststart", "-y", str(OUT)]
    subprocess.run(cmd, capture_output=True, timeout=900)

    if not OUT.exists():
        raise SystemExit("合成失败")

    probe = subprocess.run([ff, "-hide_banner", "-i", str(OUT)], capture_output=True, text=True)
    info = "\n".join(line for line in (probe.stderr or "").splitlines() if "Duration" in line or "Stream #" in line)
    print(f"\n成片：{OUT}")
    print(f"大小：{OUT.stat().st_size // 1024} KB")
    print(info)


if __name__ == "__main__":
    main()
