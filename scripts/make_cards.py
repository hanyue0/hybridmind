"""生成视频用的字卡（1920×1080 PNG）。

为什么用 HTML + 无头浏览器而不是 PIL 画字：中文字体、字重、行距、渐变这些
用 CSS 一行就能调，PIL 要手算；而且卡片风格可以和演示界面保持一致。

用法：
    python scripts/make_cards.py
输出：
    docs/video/v2/cards/*.png
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "video" / "v2" / "cards"

EDGE = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

BASE_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { width: 1920px; height: 1080px; overflow: hidden; }
body {
  background: radial-gradient(1200px 700px at 18% -10%, #16233a 0%, #0a0e14 55%);
  color: #e6edf3;
  font-family: "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;
  display: flex; flex-direction: column; justify-content: center;
  padding: 0 150px;
}
.kicker { color: #4f8cff; font-size: 26px; letter-spacing: 6px; margin-bottom: 26px; }
h1 { font-size: 92px; font-weight: 700; line-height: 1.25; letter-spacing: 1px; }
h2 { font-size: 54px; font-weight: 600; line-height: 1.4; color: #c7d5e6; }
p  { font-size: 34px; line-height: 1.85; color: #8494a7; }
.accent { color: #4f8cff; }
.green { color: #22c55e; }
.amber { color: #f5a524; }
.red { color: #ef4444; }
.mono { font-family: "Cascadia Mono", "Consolas", monospace; }
.row { display: flex; gap: 60px; align-items: baseline; }
.card-line { height: 2px; background: linear-gradient(90deg, #4f8cff, transparent); margin: 34px 0; }
.big { font-size: 150px; font-weight: 700; font-variant-numeric: tabular-nums; }
.small { font-size: 26px; color: #6b7a8c; }
table { border-collapse: collapse; font-size: 34px; margin-top: 20px; }
td, th { padding: 14px 34px; text-align: left; border-bottom: 1px solid #1e2a38; }
th { color: #8494a7; font-weight: 500; font-size: 26px; }
"""

CARDS = {
    "title": """
      <div class="kicker">2026 第九届传智杯 · AI 大模型创新应用挑战赛</div>
      <h1>HybridMind</h1>
      <div class="card-line" style="width:520px"></div>
      <h2>论文精读问答与证据锚点</h2>
      <p style="margin-top:26px">让每一条结论都能指回原文的那句话</p>
    """,
    "problem": """
      <div class="kicker">问题</div>
      <h1 style="font-size:76px">它答得很顺，<br>但你可能看不出<span class="red">错在哪</span></h1>
      <div class="row" style="margin-top:70px">
        <div><div class="big mono">44.1</div><div class="small">原文报告</div></div>
        <div style="font-size:110px;color:#3d4a5c">→</div>
        <div><div class="big mono red">41.4</div><div class="small">模型复述</div></div>
        <div style="font-size:110px;color:#3d4a5c">?</div>
        <div style="max-width:520px"><p>论文里一个数字抄错，<br>整段对比结论就失效。</p></div>
      </div>
    """,
    "arch": """
      <div class="kicker">系统</div>
      <h1 style="font-size:64px;margin-bottom:44px">解析 → 检索 → 门控路由 → 锚点</h1>
      <table>
        <tr><th>层</th><th>做什么</th><th>关键设计</th></tr>
        <tr><td>解析</td><td>PDF → 片段</td><td>页码 + 行级坐标</td></tr>
        <tr><td>检索</td><td>BM25 + 双语索引</td><td>中文问英文，命中率 91.9%</td></tr>
        <tr><td>路由</td><td>先本地、后校验</td><td>证据站得住才不上云</td></tr>
        <tr><td>锚点</td><td>答案 → 原文行</td><td>确定性对齐 + 三条校验</td></tr>
      </table>
    """,
    "result": """
      <div class="kicker">结果 · 68 道中文题</div>
      <h1 style="font-size:62px;margin-bottom:10px">更便宜，而且更能指出处</h1>
      <table style="margin-top:34px">
        <tr><th>策略</th><th>准确率</th><th>证据召回</th><th>千次成本</th></tr>
        <tr><td>全本地 1.5B</td><td>63.2%</td><td>64.5%</td><td>¥0</td></tr>
        <tr><td>全云端 DeepSeek</td><td>86.8%</td><td>91.9%</td><td>¥1.69</td></tr>
        <tr><td class="amber">证据门控路由</td><td class="amber">85.3%</td><td class="amber">93.5%</td><td class="amber">¥0.66</td></tr>
      </table>
      <p style="margin-top:36px">准确率与全云端只差 1 题，云端调用只用 39.7%</p>
    """,
    "context": """
      <div class="kicker">为什么不直接塞进长上下文</div>
      <div class="row" style="gap:110px;margin-top:30px">
        <div>
          <div class="big mono red">15000</div>
          <div class="small">tokens / 次提问 · 约 ¥0.03</div>
          <p style="margin-top:18px">论文全文塞入<br>关键信息落在中部<br>且给不出高亮坐标</p>
        </div>
        <div style="font-size:100px;color:#3d4a5c">→</div>
        <div>
          <div class="big mono green">800</div>
          <div class="small">tokens / 次提问 · 约 ¥0.0019</div>
          <p style="margin-top:18px">检索 5 个片段<br>带页码与坐标<br>答案可回指原文</p>
        </div>
      </div>
    """,
    "engineering": """
      <div class="kicker">工程</div>
      <h1 style="font-size:64px;margin-bottom:40px">一条命令起服务，所有数字可复算</h1>
      <table>
        <tr><th>项</th><th>状态</th></tr>
        <tr><td>单元测试</td><td><span class="green">46 项全绿</span>（旧 22 + 新 24）</td></tr>
        <tr><td>评测与图表</td><td>脚本生成，换判分口径可离线重判</td></tr>
        <tr><td>部署</td><td>单容器：应用 + 本地模型</td></tr>
        <tr><td>复现</td><td>下载语料 → 建索引 → 跑四组对照</td></tr>
      </table>
      <p style="margin-top:36px">没有手填的数字，也没有不可复算的结论</p>
    """,
    "closing": """
      <div class="kicker">一句话</div>
      <h1 style="font-size:78px">重要的不是答得更像，<br>而是答完之后，<br><span class="accent">你能用一秒钟验证它。</span></h1>
      <p style="margin-top:44px">HybridMind · 论文精读问答与证据锚点</p>
    """,
}


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
    browser = find_browser()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, body in CARDS.items():
        html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<style>{BASE_CSS}</style></head><body>{body}</body></html>"""
        page = OUT / f"{name}.html"
        page.write_text(html, encoding="utf-8")
        png = OUT / f"{name}.png"
        subprocess.run(
            [
                browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                "--virtual-time-budget=3000",
                "--window-size=1920,1080",
                f"--screenshot={png}", page.resolve().as_uri(),
            ],
            capture_output=True, timeout=120,
        )
        print(f"{png.name}（{png.stat().st_size // 1024 if png.exists() else 0} KB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
