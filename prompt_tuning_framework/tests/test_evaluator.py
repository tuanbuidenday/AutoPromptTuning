"""Test ③ Đánh giá — chấm điểm đúng/sai."""
from prompt_tuning_framework import Prediction, Sample
from prompt_tuning_framework.components import AccuracyEvaluator


def _preds(*outputs):
    return [Prediction(sample_id=i, output=o) for i, o in enumerate(outputs)]


def test_all_correct_scores_100(samples):
    ev = AccuracyEvaluator()
    r = ev.evaluate("p", _preds("Yes", "No", "Yes", "No"), samples)
    assert r.score == 100.0
    assert r.num_correct == 4 and r.errors == []


def test_half_correct_scores_50(samples):
    ev = AccuracyEvaluator()
    r = ev.evaluate("p", _preds("Yes", "Yes", "Yes", "Yes"), samples)
    assert r.score == 50.0
    assert len(r.errors) == 2
    assert {e.sample.id for e in r.errors} == {1, 3}


def test_case_insensitive_by_default(samples):
    ev = AccuracyEvaluator()
    r = ev.evaluate("p", _preds("yes", "NO", "YES", "no"), samples)
    assert r.score == 100.0


def test_case_sensitive_catches_case_mismatch(samples):
    ev = AccuracyEvaluator(case_sensitive=True)
    r = ev.evaluate("p", _preds("yes", "No", "Yes", "No"), samples)
    assert r.score == 75.0


def test_skips_llm_errored_samples(samples):
    """Ca gọi LLM lỗi không được tính là SAI — tránh chấm oan prompt."""
    ev = AccuracyEvaluator()
    r = ev.evaluate("p", _preds("Yes", "__ERROR__: quota", "Yes", "No"), samples)
    assert r.metadata["num_scored"] == 3
    assert r.metadata["num_skipped"] == 1
    assert r.score == 100.0  # 3/3 ca chấm được đều đúng
    assert r.results[1].correct is None


def test_skips_unlabeled_samples():
    ev = AccuracyEvaluator()
    s = [Sample(id=0, text="a", label=None), Sample(id=1, text="b", label="No")]
    r = ev.evaluate("p", _preds("Yes", "No"), s)
    assert r.metadata["num_scored"] == 1 and r.score == 100.0


def test_missing_prediction_counts_as_wrong(samples):
    ev = AccuracyEvaluator()
    r = ev.evaluate("p", [Prediction(sample_id=0, output="Yes")], samples)
    assert r.num_correct == 1
    assert len(r.errors) == 3  # 3 ca còn lại không có dự đoán -> sai
