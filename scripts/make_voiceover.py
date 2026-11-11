"""用 edge-tts 生成一版机器配音（真人配音的兜底方案）。

为什么留这条兜底：配音是视频唯一的"人工阻塞项"。如果时间不够，
一版可听的机器配音好过一个没有声音的视频；真人版到了直接替换同名文件即可，
后面的对齐、字幕、合成流程完全复用。

用法：
    python scripts/make_voiceover.py --out 配音/v2/ai --voice zh-CN-YunxiNeural
    python scripts/make_voiceover.py --list          # 只看各段字数与预计时长

语速：edge-tts 的 rate 以百分比相对基准语速。基准约 240 字/分钟，
目标是 230–250 字/分钟，因此一般用 "+0%"。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SEGMENTS = [
    (
        "seg1",
        "读一篇英文论文，最花时间的不是从头看到尾，而是回答一个具体问题："
        "这篇在哪个数据集上评的？报告的那个指标是多少？"
        "把这些丢给大模型，几秒就有答案，而且说得特别顺。"
        "但论文里最要命的就是这种顺——它把四十四点一记成四十一点四，你不翻原文根本看不出来。"
        "我自己的用法一直是先问 AI、再翻原文核对。可如果每次都要核对，我省下的时间又还回去了一半。"
        "所以我想做的不是让它答得更像人，而是让它答完之后，我能指着原文说：就是这句话。",
    ),
    (
        "seg2",
        "这是 HybridMind，一个中文论文精读助手。"
        "上面选一篇论文，我在右边用中文提问：RAG 使用多少篇 Wikipedia 文档？"
        "注意看左边：它答的是二十一个百万，也就是两千一百万。这个数字不在第一页，是系统自己检索到第四页去的。"
        "更关键的是下面这条证据锚点：S 三十，第四页。点一下，左边自动翻到第四页，把命中的那几行框出来。"
        "这条锚点不是模型自己标的。我实测过，一点五 B 的小模型经常不写出处，它把话说圆就完了。"
        "所以系统加了一层确定性对齐：用数字、英文术语、还有建库时生成的中文译文，把答案句重新钉回原文。"
        "模型守不守规矩，都不影响我核对。",
    ),
    (
        "seg3",
        "再问一个难的：Jeopardy 问题生成任务，用的测试集有多少个样本？"
        "这次本地答完，校验没通过，系统自己升级到云端重答，然后重新校验。"
        "所以它的路由逻辑不是猜这道题难不难，而是先答、再验："
        "答案里有锚点、引用编号真实存在、数字能在原文里找到，就留在本地；有一条不成立，才花钱上云。"
        "在这台没有独显的笔记本上，本地答一题要十秒左右，云端不到一秒。"
        "多等这十秒值不值，取决于本地答案的通过率——实测是百分之六十九。",
    ),
    (
        "seg4",
        "我把三篇论文、六十八道中文题跑了一遍，四种设置。"
        "全本地，准确率六十三点二，证据召回六十四点五——三分之一的回答连出处都给不出。"
        "全云端，准确率八十六点八，证据召回九十一点九，但每次问答都要花钱。"
        "然后是学习型路由，也就是我上一版用的预测本地会不会答错。"
        "它在这个场景里彻底失效了：交叉验证 AUC 只有零点六一，阈值一卡，百分之九十三的请求都要上云，等于没路由。"
        "最后是证据门控：准确率八十五点三，和全云端只差一道题；证据召回九十三点五，最高；"
        "误拒率百分之三点二，最低；云端只用掉百分之三十九点七，"
        "每千次成本从一块六毛九降到六毛六，降了六成。"
        "它没有比云端更准，但它更便宜，而且更能指出处。",
    ),
    (
        "seg5",
        "有人会问：为什么不直接把整篇论文塞进上下文？"
        "三个原因。第一，一篇二十页的论文一万五千 token，每次提问是三毛钱这个量级，比检索方案贵一个数量级。"
        "第二，已有实验证明，关键信息落在长上下文中部时，模型的表现明显下降——"
        "而论文的方法和实验，恰恰在中间。"
        "第三，也是最重要的：整篇塞进去，模型知道答案，但它给不出一个能画高亮框的坐标。"
        "检索不只是为了省钱，是为了让答案有位置。",
    ),
    (
        "seg6",
        "工程上，这个系统一条命令起服务，四十六个单元测试全绿，"
        "评测集、图表、文档全部由脚本生成，没有一个手填的数字。"
        "它当然不完美：只有三篇论文、六十八道题，准确率差一道题说明不了太多；"
        "表格跨行跨页的时候，检索还是会漏。"
        "但它至少把一件事说清楚了：让 AI 答论文，重要的不是答得更像，"
        "而是答完之后，你能用一秒钟验证它。",
    ),
]


async def synth(text: str, out: Path, voice: str, rate: str) -> None:
    """合成音频，同时写一份按词边界的 SRT（后面烧字幕直接用）。"""
    import edge_tts

    communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
    submaker = edge_tts.SubMaker()
    with out.open("wb") as fh:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                fh.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                submaker.feed(chunk)
    srt = submaker.get_srt()
    if srt:
        out.with_suffix(".srt").write_text(srt, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "配音" / "v2" / "ai"))
    ap.add_argument("--voice", default="zh-CN-YunxiNeural")
    ap.add_argument("--rate", default="+0%")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for name, text in SEGMENTS:
            print(f"{name}: {len(text)} 字，预计 {len(text)/240*60:.0f} 秒")
        print(f"合计 {sum(len(t) for _, t in SEGMENTS)} 字，"
              f"约 {sum(len(t) for _, t in SEGMENTS)/240:.1f} 分钟")
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in SEGMENTS:
        target = out / f"{name}.mp3"
        asyncio.run(synth(text, target, args.voice, args.rate))
        print(f"{target}（{len(text)} 字，{target.stat().st_size // 1024} KB）")
    (out / "segments.json").write_text(
        json.dumps({n: t for n, t in SEGMENTS}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
