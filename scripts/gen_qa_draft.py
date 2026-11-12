"""为论文 QA 评测集生成候选题目（草案）。

定位：**只出草案，不产生标准答案**。草案出来后由 ``--verify`` 逐条回到原文校验，
答案必须在对应段落里逐字出现，否则整条丢弃。判分口径仍然是确定性的，
这里用一次大模型只是为了省人工读 400 段的时间。

用法：
    python scripts/gen_qa_draft.py --paper rag_2005.11401 --limit 8
    python scripts/gen_qa_draft.py --verify data/qa_draft.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.providers import OpenAICompatProvider  # noqa: E402

BATCH = 3

PROMPT = (
    "你是论文精读的出题人。对给定的每个原文片段，出 1 道中文问题。规则：\n"
    "1. 答案必须是原文中**逐字出现**的一个数字（可带单位/百分号）或一个英文术语/短语，"
    "长度不超过 40 个字符；\n"
    "2. 问题要像学生真会问的（这篇论文用了什么数据集、某个指标是多少、某个方法怎么做），"
    "不要出现「根据上文」「该段」这类元话语；\n"
    "3. 如果该段没有可提问的确定事实，answer 填 null；\n"
    "4. 严格输出 JSON 数组，每项 {\"id\": \"S12\", \"question\": \"...\", \"answer\": \"...\"}，"
    "不要输出其它文字。"
)

# 第二轮出题：换角度，优先「需要跨句综合」的问题，补充题库用
PROMPT_CROSS = (
    "你是论文精读的出题人。对给定的每个原文片段，出 1 道中文问题。规则：\n"
    "1. 问题要**结合该段里两处以上信息**才能回答（例如某个设置的取值、某个方法的组成、"
    "两组结果的对比），而不是抄一个孤立名词；\n"
    "2. 答案必须是原文中**逐字出现**的内容，优先选：带单位的数字、百分比、英文缩写/术语；"
    "长度不超过 40 个字符；\n"
    "3. 不要出现「根据上文」「该段」这类元话语；\n"
    "4. 如果该段没有可提问的确定事实，answer 填 null；\n"
    "5. 严格输出 JSON 数组，每项 {\"id\": \"S12\", \"question\": \"...\", \"answer\": \"...\"}，"
    "不要输出其它文字。"
)

_CITE = re.compile(r"\[\d+(?:\s*[,，]\s*\d+)*\]")


def _clean(text: str) -> str:
    return _CITE.sub("", text)


def draft(provider, name: str, chunks: list[dict], start: int, limit: int,
          prompt: str = PROMPT) -> list[dict]:
    out: list[dict] = []
    batch = [c for c in chunks[start : start + limit] if len(c["text"]) >= 200]
    for i in range(0, len(batch), BATCH):
        part = batch[i : i + BATCH]
        body = "\n\n".join(f"[{c['id']}] {_clean(c['text'])}" for c in part)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": body},
        ]
        try:
            gen = provider.generate(messages, max_tokens=700, temperature=0.3)
            text = gen.text.strip()
            text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
            items = json.loads(text)
        except Exception as exc:
            print(f"  {part[0]['id']}.. 解析失败：{exc}", file=sys.stderr)
            continue
        by_id = {c["id"]: c for c in part}
        for it in items if isinstance(items, list) else []:
            cid = it.get("id")
            ans = (it.get("answer") or "").strip() if it.get("answer") else ""
            q = (it.get("question") or "").strip()
            if cid not in by_id or not ans or not q:
                continue
            out.append(
                {
                    "paper": name,
                    "chunk_id": cid,
                    "page": by_id[cid]["page"],
                    "question": q,
                    "answer": ans,
                }
            )
        print(f"  {part[0]['id']}..{part[-1]['id']} → 累计 {len(out)} 条", file=sys.stderr)
    return out


def normalize(text: str) -> str:
    text = text.lower()
    for a, b in (("\ufb01", "fi"), ("\ufb02", "fl"), ("\u00ad", "")):
        text = text.replace(a, b)
    return re.sub(r"\s+", " ", text)


def verify(path: Path, chunks_root: Path) -> list[dict]:
    """答案必须逐字出现在原段落，且不能只是引用编号或泛词。"""
    cache: dict = {}
    good: list[dict] = []
    dropped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        name = item["paper"]
        if name not in cache:
            data = json.loads((chunks_root / f"{name}.chunks.json").read_text(encoding="utf-8"))
            cache[name] = {c["id"]: c for c in data["chunks"]}
        chunk = cache[name].get(item["chunk_id"])
        if not chunk:
            dropped += 1
            continue
        ans = item["answer"]
        if len(ans) < 2 or ans.isdigit() and len(ans) <= 1:
            dropped += 1
            continue
        if normalize(ans) not in normalize(_clean(chunk["text"])):
            dropped += 1
            continue
        item["page"] = chunk["page"]
        good.append(item)
    print(f"校验通过 {len(good)} 条，丢弃 {dropped} 条", file=sys.stderr)
    return good


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=9)
    ap.add_argument("--out", default=str(ROOT / "data" / "qa_draft.jsonl"))
    ap.add_argument("--verify", default=None)
    ap.add_argument("--papers", default=str(ROOT / "data" / "papers"))
    ap.add_argument("--variant", choices=["base", "cross"], default="base",
                    help="base=第一轮出题；cross=第二轮（跨句综合角度），用于扩题库")
    args = ap.parse_args()

    if args.verify:
        good = verify(Path(args.verify), Path(args.papers))
        # 输入 qa_draft.jsonl → 输出 qa_candidates.jsonl；qa_draft_cross.jsonl → qa_candidates_cross.jsonl
        stem = Path(args.verify).stem.replace("qa_draft", "qa_candidates")
        out = Path(args.verify).with_name(f"{stem}.jsonl")
        out.write_text("\n".join(json.dumps(g, ensure_ascii=False) for g in good), encoding="utf-8")
        print(f"写入 {out}")
        return 0

    provider = OpenAICompatProvider(
        settings.cloud_base_url, settings.cloud_api_key, settings.cloud_model,
        settings.price_of(settings.cloud_model), timeout_s=120,
    )
    prompt = PROMPT_CROSS if args.variant == "cross" else PROMPT
    root = Path(args.papers)
    names = [args.paper] if args.paper else sorted(p.stem for p in root.glob("*.pdf"))
    rows: list[dict] = []
    for name in names:
        data = json.loads((root / f"{name}.chunks.json").read_text(encoding="utf-8"))
        print(f"[{name}] {len(data['chunks'])} 块，从 {args.start} 开始出 {args.limit} 块", file=sys.stderr)
        rows.extend(draft(provider, name, data["chunks"], args.start, args.limit, prompt=prompt))

    out = Path(args.out)
    mode = "a" if out.exists() else "w"
    with out.open(mode, encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"追加 {len(rows)} 条到 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
