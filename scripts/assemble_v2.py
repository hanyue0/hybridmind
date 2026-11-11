"""合成演示视频：画面（字卡 + 真实交互录屏 + 图表）配 6 段配音。

时间轴以**配音时长**为准——先量每段 mp3 的长度，再决定画面各片段放多久。
这样换配音（真人/机器）只需重跑本脚本，画面自动对齐，不用手工调时间码。

用法：
    python scripts/assemble_v2.py \
        --audio "D:/桌面/AMC-传智/配音/v2/ai" \
        --out docs/video/v2/HybridMind_演示视频_v2.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "docs" / "video" / "v2"
CARDS = VIDEO / "cards"
FIGURES = ROOT / "docs" / "figures"
GAP = 0.6  # 段间停顿

W, H, FPS = 1920, 1080, 15

# 每段画面的构成：(类型, 资源名, 时长或占比)
# None 表示"吃掉本段剩余时长"；小数表示占本段总时长的比例
PLAN = {
    "seg1": [("card", "title", 4.0), ("card", "problem", None)],
    "seg2": [("clip", "seg2.mp4", None)],
    "seg3": [("clip", "seg3.mp4", None)],
    "seg4": [("card", "result", 0.42), ("figure", "fig_paper_gate", 0.30), ("figure", "fig_paper_cost", 0.28)],
    "seg5": [("card", "context", None)],
    "seg6": [("card", "engineering", 0.55), ("card", "closing", 0.45)],
}


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise SystemExit(result.stderr.decode("utf-8", "ignore")[-1500:])


def ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_duration(path: Path) -> float:
    r = subprocess.run([ffmpeg(), "-i", str(path)], capture_output=True)
    for line in r.stderr.decode("utf-8", "ignore").splitlines():
        if "Duration:" in line:
            stamp = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = stamp.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    raise SystemExit(f"读不到时长：{path}")


def encode_still(src: Path, dur: float, out: Path) -> None:
    run([
        ffmpeg(), "-y", "-loop", "1", "-t", f"{dur:.3f}", "-i", str(src),
        "-vf", f"scale={W}:{H},setsar=1,format=yuv420p", "-r", str(FPS),
        "-c:v", "libx264", "-crf", "20", "-preset", "medium", str(out),
    ])


def encode_clip(src: Path, start: float, dur: float, out: Path) -> None:
    """截取录屏片段；源比需要的短时，用最后一帧补足（tpad）。"""
    src_len = probe_duration(src)
    take = min(dur, max(0.1, src_len - start))
    pad = max(0.0, dur - take)
    vf = f"scale={W}:{H},setsar=1,fps={FPS},format=yuv420p"
    if pad > 0.05:
        tmp = out.with_suffix(".part.mp4")
        run([
            ffmpeg(), "-y", "-ss", f"{start:.3f}", "-t", f"{take:.3f}", "-i", str(src),
            "-vf", vf, "-c:v", "libx264", "-crf", "20", "-preset", "medium", str(tmp),
        ])
        run([
            ffmpeg(), "-y", "-i", str(tmp),
            "-vf", f"tpad=stop_mode=clone:stop_duration={pad:.3f},{vf}",
            "-r", str(FPS), "-c:v", "libx264", "-crf", "20", "-preset", "medium", str(out),
        ])
        tmp.unlink(missing_ok=True)
    else:
        run([
            ffmpeg(), "-y", "-ss", f"{start:.3f}", "-t", f"{take:.3f}", "-i", str(src),
            "-vf", vf, "-c:v", "libx264", "-crf", "20", "-preset", "medium", str(out),
        ])


def build_audio(audio_dir: Path, out: Path) -> tuple[float, dict]:
    ff = ffmpeg()
    work = out.parent / "audio_parts"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    names = ["seg1", "seg2", "seg3", "seg4", "seg5", "seg6"]
    durations = {}
    lines = []
    for i, name in enumerate(names):
        src = audio_dir / f"{name}.mp3"
        wav = work / f"{name}.wav"
        run([ff, "-y", "-i", str(src), "-ar", "24000", "-ac", "1", str(wav)])
        durations[name] = probe_duration(wav)
        lines.append(f"file '{wav.resolve().as_posix()}'")
        if i < len(names) - 1:
            gap = work / f"gap{i}.wav"
            run([ff, "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", f"{GAP}", str(gap)])
            lines.append(f"file '{gap.resolve().as_posix()}'")
    listfile = work / "list.txt"
    listfile.write_text("\n".join(lines), encoding="utf-8")
    run([ff, "-y", "-f", "concat", "-safe", "0", "-i", str(listfile), "-c", "copy", str(out)])
    total = probe_duration(out)
    shutil.rmtree(work, ignore_errors=True)
    return total, durations


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", default=str(VIDEO / "HybridMind_演示视频_v2.mp4"))
    args = ap.parse_args()

    audio_dir = Path(args.audio)
    out = Path(args.out)
    work = VIDEO / "build"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    audio_path = work / "audio.wav"
    total_audio, durations = build_audio(audio_dir, audio_path)
    print(f"配音总长 {total_audio:.1f}s：" + "，".join(f"{k} {v:.1f}s" for k, v in durations.items()))

    pieces = []
    idx = 0
    names = list(PLAN)
    for order, name in enumerate(names):
        items = PLAN[name]
        seg_total = durations[name]
        fixed = sum(d for _, _, d in items if isinstance(d, float) and d >= 1.0)
        remain = max(0.5, seg_total - fixed)
        for kind, res, spec in items:
            if spec is None or (isinstance(spec, float) and spec < 1.0):
                if spec is None:
                    dur = remain
                else:
                    dur = seg_total * spec
            else:
                dur = float(spec)
            remain = max(0.0, remain - dur) if spec is None else remain

            idx += 1
            target = work / f"p{idx:02d}.mp4"
            if kind == "card":
                encode_still(CARDS / f"{res}.png", dur, target)
            elif kind == "figure":
                encode_still(FIGURES / f"{res}.png", dur, target)
            elif kind == "clip":
                encode_clip(VIDEO / res, 2.0, dur, target)
            pieces.append((target, dur))
            print(f"  {name} · {kind}:{res} {dur:.1f}s")

        # 段间停顿：配音里有 0.6s 空白，画面必须同样留白，否则整条时间轴会提前
        if order < len(names) - 1:
            idx += 1
            gap_png = work / f"gap{idx}.png"
            run([ffmpeg(), "-y", "-sseof", "-0.15", "-i", str(pieces[-1][0]),
                 "-frames:v", "1", str(gap_png)])
            gap_mp4 = work / f"p{idx:02d}.mp4"
            encode_still(gap_png, GAP, gap_mp4)
            pieces.append((gap_mp4, GAP))

    listfile = work / "list.txt"
    listfile.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p, _ in pieces), encoding="utf-8"
    )
    silent = work / "silent.mp4"
    run([ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", str(listfile), "-c", "copy", str(silent)])

    run([
        ffmpeg(), "-y", "-i", str(silent), "-i", str(audio_path),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
        "-shortest", "-movflags", "+faststart", str(out),
    ])

    print(f"\n成片：{out}")
    print(f"时长：{probe_duration(out):.1f}s")
    print(f"大小：{out.stat().st_size // 1024 // 1024} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
