"""段 4（现场演示）的实机录制。

做法：真实浏览器全屏窗口 + 真实桌面录屏 + 程序驱动鼠标与输入。
- 页面操作走 CDP（元素坐标准确、点击可靠）；
- 光标可见性靠 Win32 SetCursorPos 真移到目标位置；
- 画面用 ffmpeg gdigrab 录整个屏幕。

坐标说明：本机 PowerShell 是 DPI 非感知进程，它的坐标空间等于浏览器的 CSS 像素，
因此 CDP 拿到的 rect 可以**直接**喂给 SetCursorPos，不需要乘缩放系数。

用法：
    python scripts/record_demo.py calibrate   # 定位校准：把光标移到发送键并截图
    python scripts/record_demo.py record      # 正式录制
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import websocket

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "docs" / "video"
RAW = VIDEO / "live_raw.mp4"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
URL = "http://127.0.0.1:8000/"
DEBUG_PORT = 9222
USER_DATA = VIDEO / "edge_profile"

user32 = ctypes.windll.user32


# ---------------- Win32 ----------------

def move_cursor(x: int, y: int) -> None:
    user32.SetCursorPos(int(x), int(y))


def glide_to(x: int, y: int, steps: int = 24, duration: float = 0.65) -> None:
    """带缓动的移动，避免瞬移（瞬移一眼就能看出是程序）。"""
    pt = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    sx, sy = pt.x, pt.y
    for i in range(1, steps + 1):
        t = i / steps
        eased = 1 - (1 - t) ** 3  # ease-out cubic
        move_cursor(sx + (x - sx) * eased, sy + (y - sy) * eased)
        time.sleep(duration / steps)


def click() -> None:
    user32.mouse_event(0x0002, 0, 0, 0, 0)  # LEFTDOWN
    time.sleep(0.05)
    user32.mouse_event(0x0004, 0, 0, 0, 0)  # LEFTUP


# ---------------- CDP ----------------

class CDP:
    def __init__(self, ws_url: str) -> None:
        self.ws = websocket.create_connection(ws_url, timeout=60)
        self.seq = 0

    def call(self, method: str, params: dict | None = None) -> dict:
        self.seq += 1
        self.ws.send(json.dumps({"id": self.seq, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self.ws.recv())
            if message.get("id") == self.seq:
                if "error" in message:
                    raise RuntimeError(f"{method} 失败: {message['error']}")
                return message.get("result", {})

    def js(self, expression: str):
        result = self.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        return result.get("result", {}).get("value")

    def close(self) -> None:
        self.ws.close()


def page_ws_url() -> str:
    for _ in range(40):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{DEBUG_PORT}/json/list", timeout=2) as resp:
                targets = json.loads(resp.read())
            for target in targets:
                if target.get("type") == "page" and "127.0.0.1:8000" in target.get("url", ""):
                    return target["webSocketDebuggerUrl"]
        except Exception:
            pass
        time.sleep(0.5)
    raise SystemExit("找不到调试目标")


def center(cdp: CDP, selector: str) -> tuple[int, int]:
    js = f"""(() => {{ const el = document.querySelector({selector!r});
        if (!el) return null; const r = el.getBoundingClientRect();
        return [Math.round(r.x + r.width/2), Math.round(r.y + r.height/2)]; }})()"""
    value = cdp.js(js)
    if not value:
        raise RuntimeError(f"元素不存在: {selector}")
    return int(value[0]), int(value[1])


def type_text(cdp: CDP, text: str, delay: float = 0.055) -> None:
    """逐字输入。用 insertText 走真实输入事件，中文也能逐字出现。"""
    for char in text:
        cdp.call("Input.insertText", {"text": char})
        time.sleep(delay)


# ---------------- 录制 ----------------

def launch_browser() -> subprocess.Popen:
    # 复用同一个 profile：首次启动的同步/欢迎弹窗只在第一次出现，
    # 每次删库重来反而会重新触发它们（第三版录制就是这么被毁的）。
    USER_DATA.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [
            EDGE,
            f"--remote-debugging-port={DEBUG_PORT}",
            f"--user-data-dir={USER_DATA}",
            "--no-first-run",
            "--no-default-browser-check",
            "--start-fullscreen",
            "--disable-infobars",
            "--hide-crash-restore-bubble",
            "--disable-sync",
            "--disable-features=EdgeSyncPromo,msEdgeSyncPromo,Translate",
            "--disable-session-crashed-bubble",
            "--disable-features=Translate,MediaRouter,EdgeSyncPromo",
            URL,
        ]
    )
    return proc


def start_ffmpeg(seconds: int) -> subprocess.Popen:
    ff = __import__("imageio_ffmpeg").get_ffmpeg_exe()
    VIDEO.mkdir(parents=True, exist_ok=True)
    RAW.unlink(missing_ok=True)
    return subprocess.Popen(
        [
            ff, "-hide_banner", "-loglevel", "error",
            "-f", "gdigrab", "-framerate", "30", "-draw_mouse", "1",
            "-i", "desktop",
            "-t", str(seconds),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-y", str(RAW),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_ffmpeg(proc: subprocess.Popen) -> None:
    """给 ffmpeg 发 q 让它收尾。

    直接 terminate 会丢掉 MP4 的 moov 索引，得到无法播放的文件——
    第一次录制就是这么废掉的。
    """
    if proc.poll() is not None:
        return
    try:
        proc.stdin.write(b"q")
        proc.stdin.flush()
    except Exception:
        pass
    try:
        proc.wait(timeout=25)
    except subprocess.TimeoutExpired:
        proc.terminate()


def calibrate() -> None:
    proc = launch_browser()
    try:
        cdp = CDP(page_ws_url())
        cdp.call("Page.enable")
        time.sleep(2.0)
        info = cdp.js("""JSON.stringify({
            dpr: window.devicePixelRatio, sx: window.screenX, sy: window.screenY,
            iw: window.innerWidth, ih: window.innerHeight,
            ow: window.outerWidth, oh: window.outerHeight})""")
        print("窗口信息:", info)
        x, y = center(cdp, "#send")
        print(f"发送按钮中心（CSS px）: {x}, {y}")
        glide_to(x, y, steps=20, duration=0.8)
        time.sleep(0.4)
        ff = __import__("imageio_ffmpeg").get_ffmpeg_exe()
        shot = VIDEO / "calibration.png"
        subprocess.run(
            [ff, "-hide_banner", "-loglevel", "error", "-f", "gdigrab",
             "-framerate", "1", "-i", "desktop", "-frames:v", "1", "-y", str(shot)],
            capture_output=True, timeout=40,
        )
        print(f"校准截图: {shot}")
        cdp.close()
    finally:
        proc.terminate()


def wait_idle(cdp: CDP, timeout: float = 90.0) -> None:
    """等这一轮推理结束：发送键在生成期间是 disabled。"""
    started = time.time()
    time.sleep(0.8)
    while time.time() - started < timeout:
        if not cdp.js("document.getElementById('send').disabled"):
            return
        time.sleep(0.4)


Q_EASY = "计算 87 + 8 的值。"
Q_HARD = "某商品原价 520 元，先涨价 25%，再降价 20%（按涨价后的价格计算），最终售价是多少元？"
Q_REASON = "请证明根号 2 是无理数，并说明反证法的步骤与原理。"


def record() -> None:
    ff = __import__("imageio_ffmpeg").get_ffmpeg_exe()
    proc = launch_browser()
    recorder = None
    try:
        cdp = CDP(page_ws_url())
        cdp.call("Page.enable")
        print("页面已就绪，等待全屏提示消失…")
        time.sleep(5.0)

        move_cursor(1400, 300)
        time.sleep(1.2)

        recorder = start_ffmpeg(175)
        print("开始录制…")
        time.sleep(1.5)

        # —— 第一题：简单题走本地 ——
        x, y = center(cdp, "#input")
        glide_to(x, y, steps=26, duration=0.8)
        click()
        time.sleep(0.5)
        type_text(cdp, Q_EASY, delay=0.07)
        time.sleep(0.5)
        sx, sy = center(cdp, "#send")
        glide_to(sx, sy, steps=14, duration=0.45)
        click()
        wait_idle(cdp, timeout=60)
        time.sleep(2.0)

        bx, by = center(cdp, "#route-badge")
        glide_to(bx, by - 40, steps=20, duration=0.7)
        time.sleep(2.5)

        # —— 第二题：难题走云端 ——
        glide_to(x, y, steps=20, duration=0.7)
        click()
        time.sleep(0.4)
        type_text(cdp, Q_HARD, delay=0.045)
        time.sleep(0.4)
        glide_to(sx, sy, steps=12, duration=0.4)
        click()
        wait_idle(cdp, timeout=60)
        time.sleep(2.2)

        # —— 切强制本地，同一题再跑一次 ——
        mx, my = center(cdp, '.mode[data-mode="local"]')
        glide_to(mx, my, steps=18, duration=0.6)
        click()
        time.sleep(0.6)
        move_cursor(1400, 300)
        time.sleep(1.0)

        glide_to(x, y, steps=20, duration=0.7)
        click()
        time.sleep(0.4)
        type_text(cdp, Q_HARD, delay=0.045)
        time.sleep(0.4)
        glide_to(sx, sy, steps=12, duration=0.4)
        click()
        print("本地慢速推理中，等它答完（约 30 秒）…")
        wait_idle(cdp, timeout=120)
        time.sleep(2.5)

        stx, sty = center(cdp, ".statusbar")
        glide_to(stx + 300, sty - 4, steps=22, duration=0.8)
        time.sleep(3.0)

        # —— 第四题：推理型，云端长回答 ——
        ax, ay = center(cdp, '.mode[data-mode="auto"]')
        glide_to(ax, ay, steps=20, duration=0.7)
        click()
        time.sleep(0.8)
        glide_to(x, y, steps=20, duration=0.7)
        click()
        time.sleep(0.4)
        type_text(cdp, Q_REASON, delay=0.05)
        time.sleep(0.4)
        glide_to(sx, sy, steps=12, duration=0.4)
        click()
        wait_idle(cdp, timeout=90)
        time.sleep(4.0)

        bx2, by2 = center(cdp, "#route-badge")
        glide_to(bx2, by2 + 120, steps=18, duration=0.6)
        time.sleep(3.0)

        stop_ffmpeg(recorder)
        print("录制结束")
        cdp.close()
    finally:
        if recorder and recorder.poll() is None:
            recorder.terminate()
        proc.terminate()
        time.sleep(1)

    if RAW.exists():
        out = VIDEO / "live_demo.mp4"
        subprocess.run(
            [ff, "-hide_banner", "-loglevel", "error", "-i", str(RAW),
             "-vf", "scale=-2:1080,pad=1920:1080:(ow-iw)/2:0:color=0x0a0e14",
             "-c:v", "libx264", "-preset", "medium", "-crf", "20",
             "-pix_fmt", "yuv420p", "-y", str(out)],
            capture_output=True, timeout=600,
        )
        print(f"素材: {out}  {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "calibrate"
    if action == "calibrate":
        calibrate()
    elif action == "record":
        record()
    else:
        print("用法：calibrate | record")
