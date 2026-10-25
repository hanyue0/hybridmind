"""生成演示视频的静态卡片，并拼装成片。

分工：
- 段 1/2/3/5/6/7 用 HTML 渲染成 1920×1080 的静帧（确定性，可反复重生成）；
- 段 4 是实机操作录像（另由 scripts/record_demo.py 生成）；
- 本脚本负责把两批素材按时间轴拼成 6:30 的成片。

用法：
    python scripts/build_video.py cards     # 只渲染卡片 PNG
    python scripts/build_video.py assemble  # 拼装成片（需要段 4 素材）
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARDS = ROOT / "docs" / "video" / "cards"
FRAMES = ROOT / "docs" / "video" / "frames"
CLIPS = ROOT / "docs" / "video" / "clips"
OUT = ROOT / "docs" / "HybridMind_演示视频_无声版.mp4"
LIVE = ROOT / "docs" / "video" / "live_demo.mp4"

FFMPEG = None

BASE_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { width: 1920px; height: 1080px; overflow: hidden; }
body {
  background: radial-gradient(1400px 800px at 18% -12%, #16233a 0%, #0a0e14 58%);
  color: #e6edf3;
  font-family: "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;
  display: flex; flex-direction: column; justify-content: center;
  padding: 96px 120px; position: relative;
}
.chapter {
  position: absolute; top: 60px; left: 120px;
  font-size: 22px; letter-spacing: 3px; color: #4f8cff; font-weight: 600;
}
.chapter span { color: #55637a; margin-right: 16px; letter-spacing: 1px; }
h1 { font-size: 74px; font-weight: 700; line-height: 1.25; letter-spacing: 1px; }
h2 { font-size: 40px; font-weight: 600; color: #b8c6d6; margin-top: 18px; }
.hl { color: #4f8cff; }
.local { color: #22c55e; }
.cloud { color: #4f8cff; }
.warn { color: #f5a524; }
.row { display: flex; gap: 64px; }
.card {
  flex: 1; background: #10161f; border: 1px solid #1e2a38; border-radius: 20px;
  padding: 44px 48px;
}
.card h3 { font-size: 30px; font-weight: 600; margin-bottom: 30px; letter-spacing: 1px; }
.metric { display: flex; justify-content: space-between; align-items: baseline; padding: 20px 0; border-bottom: 1px solid #1a2532; }
.metric:last-child { border-bottom: 0; }
.metric span { font-size: 26px; color: #8494a7; }
.metric b { font-size: 44px; font-weight: 700; }
table { width: 100%; border-collapse: collapse; font-size: 30px; }
th { text-align: left; font-weight: 500; color: #8494a7; padding: 18px 20px; border-bottom: 2px solid #24313f; font-size: 24px; }
td { padding: 20px; border-bottom: 1px solid #1a2532; }
td.n { text-align: right; font-variant-numeric: tabular-nums; }
tr.mark td { background: rgba(79,140,255,.10); color: #fff; }
.note { margin-top: 40px; font-size: 24px; color: #8494a7; }
.term {
  background: #0c121a; border: 1px solid #1e2a38; border-radius: 16px;
  padding: 40px 46px; font-family: Consolas, "Courier New", monospace;
  font-size: 27px; line-height: 2.0;
}
.term .p { color: #22c55e; }
.term .c { color: #c7d4e2; }
.term .o { color: #6b7d92; }
.term .g { color: #4f8cff; }
figure { display: flex; flex-direction: column; align-items: center; padding-top: 34px; }
figure img { width: 86%; border-radius: 14px; border: 1px solid #1e2a38; }
figcaption { margin-top: 22px; font-size: 24px; color: #8494a7; }
.big3 { display: flex; flex-direction: column; gap: 34px; }
.big3 div { font-size: 46px; font-weight: 600; }
.big3 em { font-style: normal; color: #8494a7; font-size: 30px; margin-left: 20px; font-weight: 400; }
"""

ARCH = """
<div class="arch">
  <div class="arch-row"><div class="arch-box wide">④ 交互层　FastAPI + 零依赖前端（SSE 流式、路由决策面板、成本看板）</div></div>
  <div class="arch-row"><div class="arch-box wide">③ 编排层　决策 → 推理 → 兜底升级 → 记账</div></div>
  <div class="arch-row split">
    <div class="arch-box">② 路由层<br><span>预测本地答对概率 · 阈值</span></div>
    <div class="arch-box">② 账本<br><span>本地占比 · token · 费用 · 延迟</span></div>
  </div>
  <div class="arch-row split">
    <div class="arch-box local">① 本地后端　Ollama<br><span>Qwen2.5-1.5B · 零边际成本</span></div>
    <div class="arch-box cloud">① 云端后端　OpenAI 兼容<br><span>deepseek-chat · 按 token 计价</span></div>
  </div>
</div>
"""

ARCH_CSS = """
.arch { display: flex; flex-direction: column; gap: 16px; }
.arch-row { display: flex; gap: 16px; }
.arch-row.split > .arch-box { flex: 1; }
.arch-box {
  flex: 1; padding: 26px 18px; text-align: center; font-size: 25px; line-height: 1.6;
  background: #0e1620; border: 1px solid #2b3d52; border-radius: 12px; color: #d7e3f0;
}
.arch-box span { font-size: 21px; color: #7b8da3; }
.arch-box.local { border-color: #2f9e5f; background: rgba(34,197,94,.07); }
.arch-box.cloud { border-color: #3d7fd6; background: rgba(79,140,255,.07); }
"""


def card_html(body: str, extra_css: str = "") -> str:
    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<style>{BASE_CSS}{extra_css}</style></head><body>{body}</body></html>"""


def build_cards() -> list[tuple[str, str, float]]:
    """返回 [(卡片名, html, 时长秒)]。"""
    cards = []

    # 片头
    cards.append(("s0_title", card_html("""
<div style="text-align:center; display:flex; flex-direction:column; align-items:center; gap:26px">
  <div style="font-size:26px;letter-spacing:10px;color:#4f8cff">2026 第九届传智杯 · AI 大模型创新应用挑战赛</div>
  <div style="font-size:104px;font-weight:800;letter-spacing:6px;margin-top:16px">HybridMind</div>
  <div style="font-size:38px;color:#b8c6d6">端云协同的自适应模型路由</div>
  <div style="width:520px;height:2px;background:#2b3d52;margin:18px 0"></div>
  <div style="font-size:24px;color:#8494a7">B 组 · 商洛学院 · 张云迪</div>
</div>
"""), 5))

    # 段1 痛点：两个极端
    cards.append(("s1_contrast", card_html("""
<div class="chapter"><span>01</span>两难</div>
<h1>一边准三十二个点，<br>一边一分钱不花</h1>
<div class="row" style="margin-top:64px">
  <div class="card">
    <h3 class="cloud">全云端</h3>
    <div class="metric"><span>准确率</span><b class="cloud">92.5%</b></div>
    <div class="metric"><span>平均延迟</span><b class="cloud">0.75 s</b></div>
    <div class="metric"><span>千次成本</span><b class="cloud">¥0.19</b></div>
  </div>
  <div class="card">
    <h3 class="local">全本地</h3>
    <div class="metric"><span>准确率</span><b class="local">60.0%</b></div>
    <div class="metric"><span>平均延迟</span><b class="local">6.27 s</b></div>
    <div class="metric"><span>千次成本</span><b class="local">¥0</b></div>
  </div>
</div>
<div class="note">同一评测集 · 160 道封闭答案题 · 本机实测</div>
"""), 40))

    # 段2 打脸
    cards.append(("s2_reversal", card_html("""
<div class="chapter"><span>02</span>被自己的数据推翻</div>
<h1>原以为本地更快更省</h1>
<h2 style="margin-top:26px">实测：云端<span class="hl">又准又快</span>，一千次只要两毛</h2>
<table style="margin-top:56px">
  <tr><th>指标</th><th style="text-align:right">全本地</th><th style="text-align:right">全云端</th></tr>
  <tr><td>准确率</td><td class="n">60.0%</td><td class="n hl">92.5%</td></tr>
  <tr class="mark"><td>平均延迟</td><td class="n">6271 ms</td><td class="n hl">753 ms</td></tr>
  <tr class="mark"><td>千次成本</td><td class="n">¥0</td><td class="n hl">¥0.194</td></tr>
</table>
<div class="note">原因：无独显 + 单通道内存，本地推理被带宽卡在 3–4 token/s</div>
"""), 40))

    # 段3 方案
    cards.append(("s3_design", card_html(f"""
<div class="chapter"><span>03</span>换一个问法</div>
<h1 style="font-size:58px">不问「这题难不难」<br>问「<span class="hl">本地会不会答错</span>」</h1>
<div style="margin-top:46px">{ARCH}</div>
""", ARCH_CSS), 50))

    # 段5a 分布外
    cards.append(("s5_ood", card_html(f"""
<div class="chapter"><span>05</span>分布外检验</div>
<figure>
  <img src="{ (ROOT / 'docs' / 'figures' / 'fig_ood.png').as_uri() }">
  <figcaption>40 道全新句式 · 两个路由器均未重训 · 规则路由准确率与「不做路由」相同</figcaption>
</figure>
"""), 30))

    # 段5b 等预算曲线
    cards.append(("s5_curve", card_html(f"""
<div class="chapter"><span>05</span>同等预算，谁更准</div>
<figure>
  <img src="{ (ROOT / 'docs' / 'figures' / 'fig_router_budget.png').as_uri() }">
  <figcaption>同等 30% 云端预算下，学习型路由领先 12.5 个百分点</figcaption>
</figure>
"""), 30))

    # 段6 工程
    cards.append(("s6_engineer", card_html("""
<div class="chapter"><span>06</span>工程</div>
<h1 style="font-size:52px">一条命令起服务，数据全部可复算</h1>
<div class="term" style="margin-top:44px">
<div><span class="p">$</span> <span class="c">python -m app.eval.run_eval --mode cloud</span></div>
<div><span class="o">| 模式 | 样本 | 准确率 | 千次成本(元) | 平均延迟(ms) |</span></div>
<div><span class="o">| cloud | 160 | 92.5% | 0.19 | 753 |</span></div>
<div style="margin-top:14px"><span class="p">$</span> <span class="c">python -m unittest discover -s tests -t .</span></div>
<div><span class="g">Ran 22 tests in 0.047s — OK</span></div>
<div style="margin-top:14px"><span class="p">$</span> <span class="c">git log --oneline</span></div>
<div><span class="o">d1241be docs: 演示视频拍摄脚本</span></div>
<div><span class="o">a0f0675 docs: 补齐作者信息、目录页码与近年文献</span></div>
<div><span class="o">3e0c2bd docs: 技术文档（24 页 PDF）与可复现的构建管线</span></div>
</div>
"""), 40))

    # 段7 收尾
    cards.append(("s7_close", card_html("""
<div class="chapter"><span>07</span>边界</div>
<div class="big3">
  <div class="local">本地<em>隐私 · 离线 · 规模化成本</em></div>
  <div class="cloud">云端<em>准确率 · 延迟</em></div>
  <div>路由<em>在两者之间定价</em></div>
</div>
<h2 style="margin-top:70px;font-size:30px">样本 200 条，分布外仅 40 道 —— 结论有边界，但方向是清楚的。</h2>
"""), 20))

    return cards


def find_ffmpeg() -> str:
    global FFMPEG
    if FFMPEG:
        return FFMPEG
    try:
        import imageio_ffmpeg

        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        FFMPEG = shutil.which("ffmpeg")
    if not FFMPEG:
        raise SystemExit("找不到 ffmpeg")
    return FFMPEG


def find_browser() -> str:
    for path in (
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ):
        if Path(path).exists():
            return path
    raise SystemExit("找不到 Edge")


def cmd_cards() -> None:
    CARDS.mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(parents=True, exist_ok=True)
    browser = find_browser()
    for name, html, _ in build_cards():
        html_path = CARDS / f"{name}.html"
        html_path.write_text(html, encoding="utf-8")
        png = FRAMES / f"{name}.png"
        subprocess.run(
            [
                browser,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--force-device-scale-factor=1",
                "--window-size=1920,1080",
                "--virtual-time-budget=4000",
                f"--screenshot={png}",
                html_path.resolve().as_uri(),
            ],
            capture_output=True,
            timeout=120,
        )
        print(f"卡片 {name}: {'OK' if png.exists() else '失败'}")


TIMELINE = [
    ("s0_title", 5),
    ("s1_contrast", 45),
    ("s2_reversal", 45),
    ("s3_design", 55),
    ("__live__", 0),
    ("s5_ood", 38),
    ("s5_curve", 37),
    ("s6_engineer", 45),
    ("s7_close", 25),
]


def assemble() -> None:
    ff = find_ffmpeg()
    if not LIVE.exists():
        raise SystemExit(f"缺少实机素材：{LIVE}")
    CLIPS.mkdir(parents=True, exist_ok=True)

    pieces: list[Path] = []
    for name, duration in TIMELINE:
        if name == "__live__":
            pieces.append(LIVE)
            continue
        png = FRAMES / f"{name}.png"
        if not png.exists():
            raise SystemExit(f"缺少卡片：{png}")
        clip = CLIPS / f"{name}.mp4"
        clip.unlink(missing_ok=True)
        subprocess.run(
            [ff, "-hide_banner", "-loglevel", "error",
             "-loop", "1", "-i", str(png), "-t", str(duration),
             "-r", "30", "-vf", "scale=1920:1080,setsar=1",
             "-c:v", "libx264", "-preset", "medium", "-crf", "20",
             "-pix_fmt", "yuv420p", "-y", str(clip)],
            capture_output=True, timeout=180,
        )
        pieces.append(clip)
        print(f"片段 {name}: {duration}s")

    listing = CLIPS / "list.txt"
    listing.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in pieces) + "\n", encoding="utf-8"
    )
    total = 5 + 45 + 45 + 55 + 59 + 38 + 37 + 45 + 25
    subprocess.run(
        [ff, "-hide_banner", "-loglevel", "error",
         "-f", "concat", "-safe", "0", "-i", str(listing),
         "-vf", f"fade=t=in:st=0:d=0.6,fade=t=out:st={total - 0.8}:d=0.8",
         "-r", "30", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(OUT)],
        capture_output=True, timeout=1200,
    )
    if not OUT.exists():
        raise SystemExit("拼装失败")
    print(f"\n成片：{OUT}  {OUT.stat().st_size // 1024} KB  约 {total // 60}:{total % 60:02d}")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "cards"
    if action == "cards":
        cmd_cards()
    elif action == "assemble":
        assemble()
    else:
        print("用法：cards | assemble")
