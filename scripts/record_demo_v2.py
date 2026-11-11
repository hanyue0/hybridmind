"""录制演示页面的真实交互画面（CDP screencast → MP4）。

为什么不录桌面：录桌面需要人守着、还会带进任务栏与通知；
CDP 的 screencast 直接拿渲染帧，分辨率固定 1920×1080，且可以脚本化重录。

用法：
    # 段 2：本地回答 + 锚点高亮
    python scripts/record_demo_v2.py --out docs/video/v2/seg2.mp4 --seconds 42 \
        --url "http://127.0.0.1:8130/paper?paper=rag_2005.11401&jump=1" \
        --type "RAG 使用多少篇 Wikipedia 文档？"

    # 段 3：证据站不住 → 升级云端
    python scripts/record_demo_v2.py --out docs/video/v2/seg3.mp4 --seconds 50 \
        --url "http://127.0.0.1:8130/paper?paper=rag_2005.11401&jump=1" \
        --type "Jeopardy 问题生成任务用的测试集有多少个样本？"
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx
import websocket

ROOT = Path(__file__).resolve().parent.parent
EDGE = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def find_browser() -> str:
    for p in EDGE:
        if Path(p).exists():
            return p
    for name in ("msedge", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit("未找到 Edge/Chrome")


class CDP:
    def __init__(self, ws) -> None:
        self.ws = ws
        self.seq = 0
        self.frames: list[tuple[float, bytes]] = []

    def _handle(self, msg: dict) -> bool:
        """收到帧事件就落盘，并回 ack；返回是否处理过。"""
        if msg.get("method") != "Page.screencastFrame":
            return False
        p = msg["params"]
        # 注意：CDP 的 screencast timestamp 单位是「秒」，不是毫秒
        ts = float(p["metadata"].get("timestamp", time.time()))
        self.frames.append((ts, base64.b64decode(p["data"])))
        self.seq += 1
        self.ws.send(
            json.dumps(
                {"id": self.seq, "method": "Page.screencastFrameAck",
                 "params": {"sessionId": p.get("sessionId")}}
            )
        )
        return True

    def collect(self, seconds: float) -> None:
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                self.ws.settimeout(max(0.05, min(0.5, deadline - time.time())))
                msg = json.loads(self.ws.recv())
            except Exception:
                continue
            self._handle(msg)

    def send(self, method: str, params: dict | None = None) -> dict:
        self.seq += 1
        mid = self.seq
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        deadline = time.time() + 25
        while time.time() < deadline:
            try:
                self.ws.settimeout(max(0.05, deadline - time.time()))
                msg = json.loads(self.ws.recv())
            except Exception:
                break
            if self._handle(msg):
                continue
            if msg.get("id") == mid:
                return msg
        return {}


TYPING_JS = """
(function(){
  const q = %s;
  const el = document.getElementById('input');
  const send = () => document.getElementById('composer').requestSubmit();
  let i = 0;
  el.focus();
  const timer = setInterval(() => {
    el.value = q.slice(0, ++i);
    if (i >= q.length) {
      clearInterval(timer);
      setTimeout(send, 450);
    }
  }, 75);
})();
"""


def encode(frames: list[tuple[float, bytes]], out: Path, fps: int = 15) -> None:
    """把「时间戳 + JPEG」编成恒定帧率的 MP4。

    实现说明：CDP 只在画面变化时给帧，帧间隔不均匀。直接按固定 fps 编码会让
    节奏变快；用重复帧补齐到目标时长最稳。concat 的 duration 指令对图片序列
    不可靠（实测输出只有 0.68 秒），所以这里改成「按序号重复写图 + image2 读入」。
    """
    import imageio_ffmpeg

    work = out.parent / (out.stem + "_frames")
    if work.exists():
        shutil.rmtree(work)
    base = work / "seq"
    base.mkdir(parents=True, exist_ok=True)

    seq = 0
    for i, (ts, data) in enumerate(frames):
        name = work / f"f{i:05d}.jpg"
        name.write_bytes(data)
        dur = (frames[i + 1][0] - ts) if i + 1 < len(frames) else 0.5
        for _ in range(max(1, round(min(dur, 30.0) * fps))):
            target = base / f"s{seq:05d}.jpg"
            try:
                os.link(name, target)
            except OSError:
                shutil.copyfile(name, target)
            seq += 1

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y", "-framerate", str(fps), "-i", "seq/s%05d.jpg",
        "-r", str(fps), "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "20",
        "-movflags", "+faststart", str(out.resolve()),
    ]
    result = subprocess.run(cmd, cwd=str(work), capture_output=True)
    if result.returncode != 0:
        print(result.stderr.decode("utf-8", "ignore")[-1200:], flush=True)
        raise SystemExit("ffmpeg 编码失败")
    shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--type", default="")
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--type-delay", type=float, default=2.0)
    ap.add_argument("--port", type=int, default=9334)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    args = ap.parse_args()

    browser = find_browser()
    profile = Path.home() / "AppData" / "Local" / "Temp" / "hybridmind_rec_profile"
    proc = subprocess.Popen(
        [browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--mute-audio",
         f"--remote-debugging-port={args.port}", f"--user-data-dir={profile}",
         f"--window-size={args.width},{args.height}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    ws_url = None
    for _ in range(40):
        try:
            targets = httpx.get(f"http://127.0.0.1:{args.port}/json", timeout=2).json()
            pages = [t for t in targets if t.get("type") == "page"]
            if pages:
                ws_url = pages[0]["webSocketDebuggerUrl"]
                break
        except Exception:
            pass
        time.sleep(0.5)
    if not ws_url:
        proc.kill()
        raise SystemExit("连不上调试端口")

    ws = websocket.create_connection(ws_url, timeout=30, max_size=2 ** 26)
    cdp = CDP(ws)
    cdp.send("Page.enable")
    cdp.send("Emulation.setDeviceMetricsOverride", {
        "width": args.width, "height": args.height, "deviceScaleFactor": 1, "mobile": False,
    })
    cdp.send("Page.navigate", {"url": args.url})
    time.sleep(2.5)
    cdp.send("Page.startScreencast", {
        "format": "jpeg", "quality": 88, "maxWidth": args.width, "maxHeight": args.height,
        "everyNthFrame": 1,
    })

    start = time.time()
    if args.type:
        time.sleep(args.type_delay)
        cdp.send("Runtime.evaluate", {"expression": TYPING_JS % json.dumps(args.type, ensure_ascii=False)})
    cdp.collect(max(0.5, args.seconds - (time.time() - start)))
    cdp.send("Page.stopScreencast")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cdp.frames:
        print("没有收到任何帧", flush=True)
        return 1
    encode(cdp.frames, out)
    print(f"已保存 {out}（{len(cdp.frames)} 帧，{out.stat().st_size // 1024} KB）", flush=True)
    ws.close()
    proc.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
