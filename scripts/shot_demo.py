"""用 CDP 给演示页面拍一张「问答完成」的截图。

为什么不用 ``msedge --screenshot``：那条命令在页面 load 事件后立刻出图，
而这里的答案要等本地模型推理完（约 10 秒）才会出现，早拍只有空壳。
本脚本先打开真实页面、等待固定的秒数，再通过 CDP 抓屏。

用法：
    python scripts/shot_demo.py --url "http://127.0.0.1:8130/paper?paper=rag_2005.11401&ask=RAG使用多少篇Wikipedia文档"
    python scripts/shot_demo.py --out docs/figures/ui_paper_answer.png --wait 40
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import sys
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures" / "ui_paper_answer.png"))
    ap.add_argument("--wait", type=float, default=45)
    ap.add_argument("--port", type=int, default=9333)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    args = ap.parse_args()

    browser = find_browser()
    profile = Path.home() / "AppData" / "Local" / "Temp" / "hybridmind_shot_profile"
    proc = subprocess.Popen(
        [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--remote-debugging-port={args.port}",
            f"--user-data-dir={profile}",
            f"--window-size={args.width},{args.height}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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
        raise SystemExit("无法连接到浏览器的调试端口")

    ws = websocket.create_connection(ws_url, timeout=30)
    seq = 0

    def send(method: str, params: dict | None = None) -> dict:
        nonlocal seq
        seq += 1
        ws.send(json.dumps({"id": seq, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(ws.recv())
            if msg.get("id") == seq:
                return msg

    send("Page.enable")
    send("Emulation.setDeviceMetricsOverride", {
        "width": args.width, "height": args.height, "deviceScaleFactor": 1, "mobile": False,
    })
    send("Page.navigate", {"url": args.url})
    print(f"已打开 {args.url}，等待 {args.wait:.0f}s 让推理完成…")
    time.sleep(args.wait)

    shot = send("Page.captureScreenshot", {"format": "png"})
    data = shot.get("result", {}).get("data")
    if not data:
        print("截图失败", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(base64.b64decode(data))
    print(f"已保存 {out}（{out.stat().st_size // 1024} KB）")
    ws.close()
    proc.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
