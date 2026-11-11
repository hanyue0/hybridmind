"""论文问答的端到端编排：检索 → 路由（含检索特征）→ 生成 → 锚点 → 校验。

与 ``pipeline.HybridPipeline`` 的分工：
- pipeline 负责「谁答」：路由、降级、兜底、记账，一行不改地复用；
- 本模块负责「凭什么这么答」：检索、上下文、锚点、证据校验。

把两者拆开是为了保住原有的 22 项单测与评测口径——论文场景是新入口，
旧入口（/api/chat）行为完全不变，两组实验才能放在同一张表里对比。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..pipeline import HybridPipeline
from ..schemas import RouteTrace, Usage
from .anchor import Anchor, AnchorReport, extract_anchors, verify_anchors
from .parser import Paper, parse_pdf
from .prompt import build_messages
from .retriever import RetrievalHit, Retriever, build_index


@dataclass
class AnswerResult:
    question: str
    answer: str
    paper: str
    anchors: List[Anchor]
    report: AnchorReport
    hits: List[RetrievalHit]
    retrieval_features: Dict[str, float]
    trace: RouteTrace
    usage: Usage
    escalated: bool = False
    gate_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "paper": self.paper,
            "anchors": [a.to_dict() for a in self.anchors],
            "report": self.report.to_dict(),
            "retrieval": {
                "features": self.retrieval_features,
                "hits": [
                    {
                        "chunk_id": h.chunk.id,
                        "page": h.chunk.page,
                        "score": h.score,
                        "section": h.chunk.section,
                        "text": h.chunk.text,
                    }
                    for h in self.hits
                ],
            },
            "trace": {
                "route": self.trace.route,
                "difficulty": self.trace.difficulty,
                "threshold": self.trace.threshold,
                "reasons": self.trace.reasons,
                "escalated": self.trace.escalated,
                "confidence": self.trace.confidence,
                "latency_ms": self.trace.latency_ms,
                "score_kind": self.trace.score_kind,
            },
            "escalated": self.escalated,
            "gate_reason": self.gate_reason,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "cost_cny": self.usage.cost_cny,
            },
        }


class PaperStore:
    """论文库：解析结果与译文都缓存成 JSON，避免每次请求重新解析 PDF。"""

    def __init__(self, root: str | Path = "data/papers") -> None:
        self.root = Path(root)
        self._papers: Dict[str, Paper] = {}
        self._zh: Dict[str, Dict[str, str]] = {}
        self._retrievers: Dict[str, Retriever] = {}

    # ---------- 元数据 ----------
    def names(self) -> List[str]:
        if not self.root.exists():
            return []
        return sorted(p.stem for p in self.root.glob("*.pdf"))

    def meta(self) -> List[dict]:
        out = []
        for name in self.names():
            try:
                paper = self.load(name)
            except Exception:  # 损坏的 PDF 不应让整个页面挂掉
                continue
            out.append({"name": name, "pages": paper.n_pages, "chunks": len(paper.chunks)})
        return out

    # ---------- 解析缓存 ----------
    def _cache_path(self, name: str) -> Path:
        return self.root / f"{name}.chunks.json"

    def load(self, name: str) -> Paper:
        if name in self._papers:
            return self._papers[name]
        cache = self._cache_path(name)
        if cache.exists():
            paper = Paper.from_dict(json.loads(cache.read_text(encoding="utf-8")))
        else:
            paper = parse_pdf(self.root / f"{name}.pdf", name=name)
            cache.write_text(
                json.dumps(paper.to_dict(), ensure_ascii=False), encoding="utf-8"
            )
        self._papers[name] = paper
        return paper

    def zh_map(self, name: str) -> Dict[str, str]:
        if name not in self._zh:
            path = self.root / f"{name}.zh.json"
            self._zh[name] = (
                json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            )
        return self._zh[name]

    def retriever(self, name: str) -> Retriever:
        if name not in self._retrievers:
            self._retrievers[name] = build_index(self.load(name), self.zh_map(name))
        return self._retrievers[name]


class PaperQA:
    def __init__(self, pipeline: HybridPipeline, store: Optional[PaperStore] = None,
                 top_k: int = 5) -> None:
        self.pipeline = pipeline
        self.store = store or PaperStore()
        self.top_k = top_k

    def retrieve(self, paper: str, question: str, k: Optional[int] = None) -> tuple[List[RetrievalHit], Dict[str, float]]:
        retriever = self.store.retriever(paper)
        hits = retriever.search(question, k=k or self.top_k)
        return hits, retriever.features(question, hits)

    def ask(self, paper: str, question: str, mode: str = "auto",
            k: Optional[int] = None) -> AnswerResult:
        """三种模式：local / cloud / gated（auto 等同 gated）。

        **gated 是本作品的核心路由策略**：先让本地答，再拿锚点校验它——
        有锚点、编号有效、数字可溯，就认为本地可信并保留；否则升级云端重答。
        不用「预测本地会不会答错」，因为实测那条路在论文问答上 AUC 只有 0.57：
        小模型的错误是「在上下文里挑错了数字」，跟题目形态和检索质量都不相关。
        而「答案能不能被原文背书」是**可验证**的信号，判据完全确定、可复算。
        """
        hits, feats = self.retrieve(paper, question, k)
        chunks = [h.chunk for h in hits]
        zh = self.store.zh_map(paper)
        messages = build_messages(question, chunks)

        if mode == "local" or mode == "cloud":
            response = self.pipeline.answer(
                messages, mode=mode, route_text=question, rag_features=feats
            )
            anchors = extract_anchors(response.answer, chunks, zh)
            report = verify_anchors(response.answer, anchors, chunks, zh)
            return self._result(paper, question, response, anchors, report, hits, feats)

        # ---- 校验门控 ----
        local = self.pipeline.answer(
            messages, mode="local", route_text=question, rag_features=feats
        )
        anchors = extract_anchors(local.answer, chunks, zh)
        report = verify_anchors(local.answer, anchors, chunks, zh)

        if report.gate_pass:
            page = anchors[0].page if anchors else 0
            reason = (
                f"本地答案通过证据校验（锚点 {anchors[0].chunk_id} · 第 {page} 页，"
                f"引用编号有效）→ 保留本地结果"
            )
            trace = local.trace
            trace.reasons = list(trace.reasons) + [reason]
            return self._result(
                paper, question, local, anchors, report, hits, feats,
                escalated=False, gate_reason=reason,
            )

        why = []
        if not anchors:
            why.append("本地答案找不到可对齐的原文")
        if report.invalid_refs:
            why.append(f"引用编号无效 {','.join(report.invalid_refs)}")
        if report.missing_numbers:
            why.append(f"数字不可溯 {','.join(report.missing_numbers[:3])}")
        reason = "本地校验未通过（" + "；".join(why or ["锚点不足"]) + "）→ 升级云端重答"

        cloud = self.pipeline.answer(
            messages, mode="cloud", route_text=question, rag_features=feats
        )
        anchors = extract_anchors(cloud.answer, chunks, zh)
        report = verify_anchors(cloud.answer, anchors, chunks, zh)
        trace = RouteTrace(
            route="cloud",
            difficulty=local.trace.difficulty,
            threshold=local.trace.threshold,
            reasons=list(local.trace.reasons) + [reason, "云端已重答并重新校验"],
            escalated=True,
            confidence=None,
            latency_ms=round(local.trace.latency_ms + cloud.trace.latency_ms, 1),
            score_kind=local.trace.score_kind,
        )
        usage = Usage(
            prompt_tokens=local.usage.prompt_tokens + cloud.usage.prompt_tokens,
            completion_tokens=local.usage.completion_tokens + cloud.usage.completion_tokens,
            cost_cny=round(local.usage.cost_cny + cloud.usage.cost_cny, 6),
        )
        return self._result(
            paper, question, cloud, anchors, report, hits, feats,
            escalated=True, gate_reason=reason, trace=trace, usage=usage,
        )

    @staticmethod
    def _result(paper: str, question: str, response, anchors, report, hits, feats,
                escalated: bool = False, gate_reason: str = "",
                trace=None, usage=None) -> AnswerResult:
        return AnswerResult(
            question=question,
            answer=response.answer,
            paper=paper,
            anchors=anchors,
            report=report,
            hits=hits,
            retrieval_features=feats,
            trace=trace or response.trace,
            usage=usage or response.usage,
            escalated=escalated,
            gate_reason=gate_reason,
        )

    # ---------- 流式（演示界面） ----------

    def ask_stream(self, paper: str, question: str, mode: str = "auto",
                   k: Optional[int] = None):
        """流式版本的门控：先流式出本地答案，校验不过就用云端答案替换。

        事件类型：retrieval / trace / delta / status / escalate / reset / done / error。
        「reset」告诉前端清空已显示的本地答案——这是流式场景下做"升级重答"
        必须的：本地已经吐了一半的字，不能让它和云端答案拼在一起。
        """
        hits, feats = self.retrieve(paper, question, k)
        chunks = [h.chunk for h in hits]
        zh = self.store.zh_map(paper)
        messages = build_messages(question, chunks)

        yield {
            "type": "retrieval",
            "features": feats,
            "hits": [
                {
                    "chunk_id": h.chunk.id,
                    "page": h.chunk.page,
                    "score": h.score,
                    "section": h.chunk.section,
                    "text": h.chunk.text,
                }
                for h in hits
            ],
        }

        if mode in ("local", "cloud"):
            for ev in self.pipeline.answer_stream(
                messages, mode=mode, route_text=question, rag_features=feats
            ):
                if ev.get("type") == "done":
                    anchors = extract_anchors(ev.get("answer", ""), chunks, zh)
                    report = verify_anchors(ev.get("answer", ""), anchors, chunks, zh)
                    ev["anchors"] = [a.to_dict() for a in anchors]
                    ev["report"] = report.to_dict()
                    ev["escalated"] = False
                    ev["gate_reason"] = ""
                yield ev
            return

        # ---- 门控：先本地 ----
        local_text = ""
        local_trace: Optional[dict] = None
        local_usage: Optional[dict] = None
        for ev in self.pipeline.answer_stream(
            messages, mode="local", route_text=question, rag_features=feats
        ):
            etype = ev.get("type")
            if etype == "delta":
                local_text += ev.get("text", "")
                yield ev
            elif etype == "done":
                local_trace = ev.get("trace")
                local_usage = ev.get("usage")
            elif etype == "trace":
                yield ev
            elif etype == "error":
                yield ev
                return

        anchors = extract_anchors(local_text, chunks, zh)
        report = verify_anchors(local_text, anchors, chunks, zh)

        if report.gate_pass:
            page = anchors[0].page if anchors else 0
            reason = (
                f"本地答案通过证据校验（锚点 {anchors[0].chunk_id} · 第 {page} 页，"
                f"引用编号有效）→ 保留本地结果"
            )
            trace = dict(local_trace or {})
            trace["reasons"] = list(trace.get("reasons", [])) + [reason]
            yield {
                "type": "done",
                "answer": local_text.strip(),
                "model": "local",
                "trace": trace,
                "usage": local_usage or {},
                "anchors": [a.to_dict() for a in anchors],
                "report": report.to_dict(),
                "escalated": False,
                "gate_reason": reason,
            }
            return

        why = []
        if not anchors:
            why.append("本地答案找不到可对齐的原文")
        if report.invalid_refs:
            why.append(f"引用编号无效 {','.join(report.invalid_refs)}")
        if report.missing_numbers:
            why.append(f"数字不可溯 {','.join(report.missing_numbers[:3])}")
        reason = "本地校验未通过（" + "；".join(why or ["锚点不足"]) + "）→ 升级云端重答"
        yield {"type": "escalate", "text": reason}
        yield {"type": "reset"}

        cloud_text = ""
        cloud_trace: Optional[dict] = None
        cloud_usage: Optional[dict] = None
        for ev in self.pipeline.answer_stream(
            messages, mode="cloud", route_text=question, rag_features=feats
        ):
            etype = ev.get("type")
            if etype == "delta":
                cloud_text += ev.get("text", "")
                yield ev
            elif etype == "done":
                cloud_trace = ev.get("trace")
                cloud_usage = ev.get("usage")
                cloud_text = ev.get("answer", cloud_text)
            elif etype == "error":
                yield ev
                return

        anchors = extract_anchors(cloud_text, chunks, zh)
        report = verify_anchors(cloud_text, anchors, chunks, zh)
        lt = local_trace or {}
        ct = cloud_trace or {}
        trace = {
            "route": "cloud",
            "difficulty": lt.get("difficulty", 0.0),
            "threshold": lt.get("threshold", 0.0),
            "reasons": list(lt.get("reasons", [])) + [reason, "云端已重答并重新校验"],
            "escalated": True,
            "confidence": None,
            "latency_ms": round(float(lt.get("latency_ms", 0.0)) + float(ct.get("latency_ms", 0.0)), 1),
            "score_kind": lt.get("score_kind", "difficulty"),
        }
        lu = local_usage or {}
        cu = cloud_usage or {}
        usage = {
            "prompt_tokens": int(lu.get("prompt_tokens", 0)) + int(cu.get("prompt_tokens", 0)),
            "completion_tokens": int(lu.get("completion_tokens", 0)) + int(cu.get("completion_tokens", 0)),
            "cost_cny": round(float(lu.get("cost_cny", 0.0)) + float(cu.get("cost_cny", 0.0)), 6),
        }
        yield {
            "type": "done",
            "answer": cloud_text.strip(),
            "model": "cloud",
            "trace": trace,
            "usage": usage,
            "anchors": [a.to_dict() for a in anchors],
            "report": report.to_dict(),
            "escalated": True,
            "gate_reason": reason,
        }
