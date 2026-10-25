"""ModelScope 创空间入口（Gradio SDK）。

创空间默认模板是 Gradio，直接在根目录放 app.py 即可运行，不需要改 SDK 设置。
界面比本地那版朴素，但把「路由决策」完整展示出来了——这是演示的核心。

本地调试：
    python app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from app.config import settings
from app.pipeline import HybridPipeline, Ledger
from app.providers import ProviderError, build_local_provider, OpenAICompatProvider

ledger = Ledger()
pipeline = HybridPipeline(
    local=build_local_provider(settings),
    cloud=OpenAICompatProvider(
        settings.cloud_base_url,
        settings.cloud_api_key,
        settings.cloud_model,
        settings.price_of(settings.cloud_model),
        timeout_s=settings.timeout_s,
    ),
    settings=settings,
    ledger=ledger,
)

EXAMPLES = [
    ["计算 87 + 8 的值。"],
    ["某商品原价 520 元，先涨价 25%，再降价 20%（按涨价后的价格计算），最终售价是多少元？"],
    ["请证明根号 2 是无理数，并说明反证法的步骤与原理。"],
]


def run(question: str, mode_label: str):
    if not question.strip():
        return "（请输入问题）", "", ""

    mode = {"自动路由": "auto", "强制本地": "local", "强制云端": "cloud"}.get(mode_label, "auto")
    try:
        result = pipeline.answer([{"role": "user", "content": question}], mode=mode)
    except ProviderError as exc:
        tip = "\n\n（提示：若提示未配置密钥，请见页面顶部说明）" if not settings.cloud_ready else ""
        return f"推理失败：{exc}{tip}", "", ""
    except Exception as exc:
        return f"服务异常：{type(exc).__name__}: {exc}", "", ""

    trace = result.trace
    route_cn = "本地 LOCAL" if trace.route == "local" else "云端 CLOUD"
    badge = f"**{route_cn}**" + ("（已升级）" if trace.escalated else "")

    trace_md = "\n".join(f"- {reason}" for reason in trace.reasons)
    trace_md += (
        f"\n\n---\n\n"
        f"- 分值 / 阈值：**{trace.difficulty:.2f} / {trace.threshold:.2f}**"
        f"（{'本地答对概率' if trace.score_kind == 'p_local_correct' else '难度分'}）\n"
        f"- 延迟：**{trace.latency_ms:.0f} ms**\n"
        f"- 费用：**¥{result.usage.cost_cny:.6f}**\n"
        f"- tokens：{result.usage.prompt_tokens} 入 / {result.usage.completion_tokens} 出"
    )

    stats = ledger.snapshot()
    stats_md = (
        f"请求 {stats['requests']} 次 · 本地占比 {stats['local_ratio']:.0%} · "
        f"累计 ¥{stats['cost_cny']:.6f} · 升级 {stats['escalations']} 次"
    )
    return result.answer, f"{badge}\n\n{trace_md}", stats_md


with gr.Blocks(title="HybridMind") as demo:
    if not settings.cloud_ready:
        _hint = "> ⚠️ **本实例未配置云端密钥**：云端一档不可用。请在空间设置中注入 `CLOUD_API_KEY` 环境变量。"
    elif settings.local_backend == "ollama":
        _hint = ("> ℹ️ **本地模型未在本实例部署**（Gradio 空间无 Ollama 服务）。"
                 "本演示展示的是云端路径与路由决策逻辑；本地一档在完整部署中可用。")
    else:
        _hint = "> ✅ 本地与云端两档均可用。"

    gr.Markdown(
        "# HybridMind · 端云协同的自适应模型路由\n"
        "预测「本地模型能否答对这道题」：判得准留在本地（零费用），判不准才升级云端。\n\n"
        f"当前策略：`{settings.router_mode}` · 本地后端：`{settings.local_backend}`\n\n"
        + _hint
    )
    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(label="提问", lines=2, placeholder="输入一个问题…")
            mode = gr.Radio(["自动路由", "强制本地", "强制云端"], value="自动路由", label="路由模式")
            submit = gr.Button("发送", variant="primary")
            answer = gr.Markdown(label="回答")
            gr.Examples(examples=EXAMPLES, inputs=question)
        with gr.Column(scale=2):
            gr.Markdown("### 路由决策")
            trace = gr.Markdown()
            gr.Markdown("### 账本")
            stats = gr.Markdown()

    submit.click(run, inputs=[question, mode], outputs=[answer, trace, stats])
    question.submit(run, inputs=[question, mode], outputs=[answer, trace, stats])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
