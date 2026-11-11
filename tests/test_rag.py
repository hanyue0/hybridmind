"""论文精读层的单元测试。

原则与 test_core 一致：不联网、不依赖真实 PDF、不依赖云端密钥——
评委 clone 下来一条命令就能跑，测试要能在任何机器上秒过。
"""

from __future__ import annotations

import unittest

import pymupdf

from app.eval import metrics_rag
from app.rag.anchor import extract_anchors, is_refusal, verify_anchors
from app.rag.parser import Chunk, parse_pdf
from app.rag.retriever import Retriever, tokenize, tokenize_en, tokenize_zh
from app.router.classifier import FEATURE_NAMES, LocalCorrectnessModel, RAG_FEATURE_NAMES


def _make_pdf(tmpdir: str) -> str:
    """造一份结构可控的小 PDF：一个标题 + 两段正文。"""
    path = f"{tmpdir}/fake_paper.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "1 Introduction", fontsize=15)
    body = (
        "Retrieval augmented generation combines a parametric memory with a "
        "non parametric index of Wikipedia. The retriever returns documents "
        "and the generator conditions on them jointly. "
    ) * 3
    y = 100
    for line in body.split(". "):
        page.insert_text((72, y), line.strip() + ".", fontsize=9)
        y += 12
    doc.save(path)
    doc.close()
    return path


class ParserTest(unittest.TestCase):
    def test_parses_into_chunks_with_page_and_bbox(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            paper = parse_pdf(_make_pdf(tmp), min_chars=60)
        self.assertGreaterEqual(len(paper.chunks), 1)
        chunk = paper.chunks[0]
        self.assertEqual(chunk.page, 1)
        self.assertTrue(chunk.text)
        self.assertEqual(len(chunk.bbox), 4)
        self.assertGreater(chunk.bbox[2], chunk.bbox[0])
        self.assertTrue(chunk.lines, "必须保留行级坐标，否则无法精确高亮")

    def test_chunk_ids_are_sequential(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            paper = parse_pdf(_make_pdf(tmp), min_chars=40)
        ids = [c.id for c in paper.chunks]
        self.assertEqual(ids, [f"S{i}" for i in range(1, len(ids) + 1)])

    def test_roundtrip_dict(self):
        chunk = Chunk(id="S1", page=2, text="hello world", bbox=(1, 2, 3, 4), section="Intro")
        again = Chunk.from_dict(chunk.to_dict())
        self.assertEqual(again.id, chunk.id)
        self.assertEqual(again.bbox, chunk.bbox)
        self.assertEqual(again.page, 2)


class TokenizerTest(unittest.TestCase):
    def test_english_stemming_unifies_plural(self):
        self.assertEqual(tokenize_en("documents document"), ["document", "document"])

    def test_chinese_bigrams(self):
        self.assertIn("检索", tokenize_zh("检索增强生成"))

    def test_stopwords_dropped(self):
        self.assertNotIn("the", tokenize("the retrieval"))

    def test_mixed_language(self):
        toks = tokenize("RAG 的检索器")
        self.assertIn("rag", toks)
        self.assertIn("检索", toks)


class RetrieverTest(unittest.TestCase):
    def _retriever(self) -> Retriever:
        chunks = [
            Chunk("S1", 1, "The model uses a Wikipedia index of 21M documents.", bbox=(0, 0, 10, 10)),
            Chunk("S2", 2, "We fine-tune BART with Adam for 10 steps.", bbox=(0, 0, 10, 10)),
            Chunk("S3", 3, "Unrelated paragraph about decoding beams and scores.", bbox=(0, 0, 10, 10)),
        ]
        return Retriever(chunks)

    def test_ranks_matching_chunk_first(self):
        hits = self._retriever().search("how many documents in the Wikipedia index", k=3)
        self.assertEqual(hits[0].chunk.id, "S1")

    def test_chinese_query_hits_translation_index(self):
        r = self._retriever()
        r.zh = {"S2": "我们用 Adam 微调 BART"}
        hits = Retriever(r.chunks, zh=r.zh).search("用什么优化器微调", k=2)
        self.assertEqual(hits[0].chunk.id, "S2")

    def test_features_shape_and_range(self):
        r = self._retriever()
        hits = r.search("Wikipedia documents", k=3)
        feats = r.features("Wikipedia documents", hits)
        self.assertEqual(set(feats), {"top1_score", "score_gap", "n_pages_hit", "coverage"})
        self.assertGreater(feats["top1_score"], 0.0)
        self.assertLessEqual(feats["top1_score"], 1.0)
        self.assertGreaterEqual(feats["coverage"], 0.0)

    def test_empty_query_returns_nothing(self):
        self.assertEqual(self._retriever().search("", k=3), [])


class AnchorTest(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            Chunk("S1", 1, "RAG combines parametric memory with a non-parametric index.", bbox=(0, 0, 5, 5),
                  lines=[{"t": "RAG combines parametric memory", "b": [0, 0, 5, 5]}]),
            Chunk("S2", 5, "RAG-Token achieves an EM of 44.1 on Natural Questions.", bbox=(6, 6, 9, 9),
                  lines=[{"t": "RAG-Token achieves an EM of 44.1", "b": [6, 6, 9, 9]}]),
        ]

    def test_explicit_citation_is_used(self):
        anchors = extract_anchors("在 NQ 上 EM 是 44.1 [S2]。", self.chunks)
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0].chunk_id, "S2")
        self.assertEqual(anchors[0].method, "cited")
        self.assertEqual(anchors[0].page, 5)
        self.assertIn("44.1", anchors[0].quote)

    def test_fallback_alignment_when_model_omits_citation(self):
        anchors = extract_anchors("该模型在 Natural Questions 上的 EM 是 44.1。", self.chunks)
        self.assertTrue(anchors, "模型不标编号时必须能兜底对齐")
        self.assertEqual(anchors[0].chunk_id, "S2")
        self.assertEqual(anchors[0].method, "aligned")

    def test_invalid_reference_is_flagged(self):
        anchors = extract_anchors("结论 [S9]。", self.chunks)
        report = verify_anchors("结论 [S9]。", anchors, self.chunks)
        self.assertIn("S9", report.invalid_refs)
        self.assertFalse(report.citation_valid)

    def test_number_not_in_cited_chunk_is_flagged(self):
        anchors = extract_anchors("准确率是 99.9 [S2]。", self.chunks)
        report = verify_anchors("准确率是 99.9 [S2]。", anchors, self.chunks)
        self.assertIn("99.9", report.missing_numbers)
        self.assertFalse(report.valid)

    def test_valid_case_passes_all_rules(self):
        anchors = extract_anchors("EM 是 44.1 [S2]。", self.chunks)
        report = verify_anchors("EM 是 44.1 [S2]。", anchors, self.chunks)
        self.assertTrue(report.valid)

    def test_citation_ids_not_mistaken_for_terms(self):
        """[S2] 里的 S2 不能被当成英文术语——否则每条引用都会被判术语不一致。"""
        anchors = extract_anchors("EM 是 44.1 [S2]。", self.chunks)
        report = verify_anchors("EM 是 44.1 [S2]。", anchors, self.chunks)
        self.assertEqual(report.missing_terms, [])

    def test_refusal_detected(self):
        self.assertTrue(is_refusal("原文未提及"))
        self.assertFalse(is_refusal("EM 是 44.1"))


class RagMetricsTest(unittest.TestCase):
    def test_citations_stripped_before_numeric_scoring(self):
        # 不剥离 [S3] 的话，最后一个数字会变成 3，正确答案被误判
        self.assertTrue(metrics_rag.answer_correct("EM 是 44.1 [S3]", "44.1"))
        self.assertFalse(metrics_rag.answer_correct("EM 是 44.2 [S3]", "44.1"))

    def test_evidence_hit_checks_gold_pages(self):
        self.assertTrue(metrics_rag.evidence_hit([5, 6], [5]))
        self.assertFalse(metrics_rag.evidence_hit([7], [5]))
        self.assertFalse(metrics_rag.evidence_hit([], [5]))

    def test_summary_handles_empty(self):
        self.assertEqual(metrics_rag.summarize([]), {})

    def test_summary_computes_rates(self):
        records = [
            {"type": "number", "correct": True, "evidence_hit": True, "citation_valid": True,
             "number_ok": True, "term_ok": True, "n_anchors": 1, "refused": False,
             "route": "local", "cost_cny": 0.0, "latency_ms": 100},
            {"type": "number", "correct": False, "evidence_hit": False, "citation_valid": False,
             "number_ok": False, "term_ok": True, "n_anchors": 0, "refused": False,
             "route": "cloud", "cost_cny": 0.001, "latency_ms": 300},
        ]
        s = metrics_rag.summarize(records)
        self.assertEqual(s["n"], 2)
        self.assertAlmostEqual(s["accuracy"], 0.5)
        self.assertAlmostEqual(s["evidence_recall"], 0.5)
        self.assertAlmostEqual(s["cloud_ratio"], 0.5)
        self.assertAlmostEqual(s["anchored_rate"], 0.5)


class RouterCompatTest(unittest.TestCase):
    def test_extra_features_ignored_by_14_dim_model(self):
        """旧模型（14 维）传检索特征时必须给出完全相同的概率——算术集结果不能变。"""
        model = LocalCorrectnessModel(
            feature_names=list(FEATURE_NAMES),
            mean=[0.0] * len(FEATURE_NAMES),
            scale=[1.0] * len(FEATURE_NAMES),
            coef=[0.01] * len(FEATURE_NAMES),
            intercept=0.0,
        )
        text = "先算 33 乘以 7，再加 11"
        base = model.predict_proba(text)
        with_rag = model.predict_proba(text, {"top1_score": 0.1, "coverage": 0.2})
        self.assertAlmostEqual(base, with_rag)

    def test_rag_features_reach_18_dim_model(self):
        names = list(FEATURE_NAMES) + list(RAG_FEATURE_NAMES)
        model = LocalCorrectnessModel(
            feature_names=names,
            mean=[0.0] * len(names),
            scale=[1.0] * len(names),
            coef=[0.0] * len(FEATURE_NAMES) + [10.0, 0.0, 0.0, 0.0],
            intercept=0.0,
        )
        low = model.predict_proba("问题", {"top1_score": 0.0})
        high = model.predict_proba("问题", {"top1_score": 1.0})
        self.assertGreater(high, low)


class PaperStoreTest(unittest.TestCase):
    """上传的论文放在 uploads/ 子目录里，必须和内置论文一样能被找到。"""

    def test_resolves_builtin_and_uploaded(self):
        import json
        import tempfile
        from pathlib import Path

        from app.rag.qa import PaperStore

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "uploads").mkdir()
            paper = {
                "name": "uploaded",
                "path": "",
                "n_pages": 1,
                "chunks": [
                    {"id": "S1", "page": 1, "text": "hello world", "bbox": [0, 0, 1, 1],
                     "section": "", "lines": []}
                ],
            }
            (root / "uploads" / "uploaded.chunks.json").write_text(
                json.dumps(paper), encoding="utf-8"
            )
            # 列表以 PDF 为准（页面渲染需要原始文件），这里放个占位文件
            (root / "uploads" / "uploaded.pdf").write_bytes(b"%PDF-1.4\n")
            (root / "uploads" / "uploaded.zh.json").write_text(
                json.dumps({"S1": "你好世界"}), encoding="utf-8"
            )
            store = PaperStore(root)
            self.assertIn("uploaded", store.names())
            self.assertEqual(len(store.load("uploaded").chunks), 1)
            self.assertEqual(store.zh_map("uploaded")["S1"], "你好世界")

    def test_missing_paper_raises(self):
        import tempfile
        from pathlib import Path

        from app.rag.qa import PaperStore

        with tempfile.TemporaryDirectory() as tmp:
            store = PaperStore(Path(tmp))
            with self.assertRaises(FileNotFoundError):
                store.load("nope")


if __name__ == "__main__":
    unittest.main()
