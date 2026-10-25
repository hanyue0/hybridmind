"""路由与判分的单元测试——保证「阈值一变，行为可预期」。"""

from __future__ import annotations

import unittest

from app.eval import metrics
from app.router import build_decider, confidence, features, policy


class FeaturesTest(unittest.TestCase):
    def test_simple_question_scores_low(self) -> None:
        score, _ = features.difficulty(features.extract("1+1 等于几？"))
        self.assertLess(score, 0.45)

    def test_reasoning_question_scores_high(self) -> None:
        text = "请证明根号2是无理数，并说明反证法的步骤与原理。"
        score, reasons = features.difficulty(features.extract(text))
        self.assertGreaterEqual(score, 0.45)
        self.assertTrue(reasons)

    def test_long_text_increases_score(self) -> None:
        short, _ = features.difficulty(features.extract("你好"))
        long, _ = features.difficulty(features.extract("你好" * 200))
        self.assertGreater(long, short)


class PolicyTest(unittest.TestCase):
    def test_route_respects_threshold(self) -> None:
        decision = policy.decide("1+1 等于几？", theta=0.45)
        self.assertEqual(decision.route, "local")

    def test_force_overrides_threshold(self) -> None:
        decision = policy.decide("1+1 等于几？", theta=0.45, force="cloud")
        self.assertEqual(decision.route, "cloud")
        self.assertTrue(any("手动指定" in r for r in decision.reasons))


class ConfidenceTest(unittest.TestCase):
    def test_parses_plain_number(self) -> None:
        self.assertAlmostEqual(confidence.parse_confidence("0.82"), 0.82)

    def test_parses_number_with_noise(self) -> None:
        self.assertAlmostEqual(confidence.parse_confidence("置信度约为 0.7"), 0.7)

    def test_clamps_out_of_range(self) -> None:
        self.assertAlmostEqual(confidence.parse_confidence("1"), 1.0)

    def test_returns_none_when_no_number(self) -> None:
        self.assertIsNone(confidence.parse_confidence("我很有把握"))


class MetricsTest(unittest.TestCase):
    def test_numeric_last_number_wins(self) -> None:
        self.assertTrue(metrics.is_correct("先算 3×4=12，答案是 15", "15"))
    def test_numeric_mismatch(self) -> None:
        self.assertFalse(metrics.is_correct("答案是 14", "15"))

    def test_float_tolerance(self) -> None:
        self.assertTrue(metrics.is_correct("最终售价 88.20 元", "88.2"))

    def test_short_answer_alias(self) -> None:
        self.assertTrue(metrics.is_correct("珠穆朗玛峰", "珠穆朗玛峰", ["珠峰"]))

    def test_short_answer_negative(self) -> None:
        self.assertFalse(metrics.is_correct("黄山", "珠穆朗玛峰"))

    def test_percentile(self) -> None:
        self.assertAlmostEqual(metrics.percentile([1, 2, 3, 4], 0.5), 2.5)

    def test_remainder_clause_is_stripped(self) -> None:
        """问商答「商 18，余数 4」应判对：余数是题面外的附带量。"""
        self.assertTrue(metrics.is_correct("商是 18，余数是 4。", "18"))

    def test_wrong_quotient_still_fails(self) -> None:
        """剥离余数不能掩盖商本身算错。"""
        self.assertFalse(metrics.is_correct("商是 20，余数是 7。", "21"))


class FakeModel:
    """替身：固定阈值 0.5，简单题返回高概率。"""

    threshold = 0.5

    def predict_proba(self, text: str) -> float:
        return 0.9 if "简单" in text else 0.1


class LearnedPolicyTest(unittest.TestCase):
    def test_high_probability_stays_local(self) -> None:
        decision = policy.decide_learned("这是一道简单题", FakeModel())
        self.assertEqual(decision.route, "local")
        self.assertEqual(decision.score_kind, "p_local_correct")

    def test_low_probability_goes_cloud(self) -> None:
        decision = policy.decide_learned("一道复杂的工程问题", FakeModel())
        self.assertEqual(decision.route, "cloud")

    def test_force_overrides_probability(self) -> None:
        decision = policy.decide_learned("一道复杂的工程问题", FakeModel(), force="local")
        self.assertEqual(decision.route, "local")

    def test_reasons_expose_both_views(self) -> None:
        decision = policy.decide_learned("一道复杂的工程问题", FakeModel())
        joined = " ".join(decision.reasons)
        self.assertIn("本地答对概率", joined)
        self.assertIn("参考难度分", joined)


class DeciderFactoryTest(unittest.TestCase):
    def test_missing_model_falls_back_to_rule(self) -> None:
        class S:
            router_mode = "learned"
            router_model_path = "data/__not_exist__.json"
            theta_local = 0.45

        decider = build_decider(S())
        decision = decider("1+1 等于几？", None)
        self.assertEqual(decision.score_kind, "difficulty")


if __name__ == "__main__":
    unittest.main()
