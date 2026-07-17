"""Kiểm thử các phương pháp đo lường: khoảng tin cậy, kiểm định, độ dài, đa model.

Điều các test này canh giữ: một con số điểm trần là thứ gây hiểu nhầm. Framework
phải nói được "tôi không chắc tới mức này" thay vì chỉ đưa ra 100.0.
"""
import math

import pytest

from prompt_tuning_framework.components.evaluators import (AccuracyEvaluator,
                                                           CompositeEvaluator,
                                                           CrossModelEvaluator,
                                                           count_words)
from prompt_tuning_framework.core.stats import (discordant_counts,
                                                mcnemar_exact,
                                                min_flips_for_significance,
                                                non_inferiority,
                                                wilson_interval)
from prompt_tuning_framework.core.types import EvalResult, Prediction, Sample


# ---------- khoảng tin cậy Wilson ---------------------------------------
def test_wilson_khong_bao_gio_tuyen_bo_chac_chan_tuyet_doi():
    """16/16 đúng KHÔNG có nghĩa là chắc chắn 100%.

    Đây là lý do dùng Wilson thay vì Wald: Wald cho ra [100, 100] ở trường hợp
    này, tức tuyên bố chắc nịch chỉ từ 16 mẫu.
    """
    lo, hi = wilson_interval(16, 16)
    assert hi == 100.0
    assert 80 < lo < 81      # ~80.6
    assert lo < 100, "16 mẫu không đủ để loại trừ khả năng prompt chỉ đúng ~81%"


def test_wilson_khop_gia_tri_tinh_tay():
    for x, n, exp_lo, exp_hi in [
        (16, 16, 80.6, 100.0),
        (15, 16, 71.7, 98.9),
        (11, 16, 44.4, 85.8),
    ]:
        lo, hi = wilson_interval(x, n)
        assert lo == pytest.approx(exp_lo, abs=0.1)
        assert hi == pytest.approx(exp_hi, abs=0.1)


def test_wilson_cang_nhieu_mau_khoang_cang_hep():
    rong = [wilson_interval(round(0.9 * n), n) for n in (16, 100, 400)]
    do_rong = [hi - lo for lo, hi in rong]
    assert do_rong[0] > do_rong[1] > do_rong[2]


def test_wilson_bien():
    assert wilson_interval(0, 0) == (0.0, 0.0)      # không có mẫu -> không kết luận
    lo, _ = wilson_interval(0, 16)
    assert lo == 0.0
    with pytest.raises(ValueError):
        wilson_interval(17, 16)


# ---------- McNemar ------------------------------------------------------
def test_mcnemar_ket_qua_headline_chua_du_y_nghia():
    """68.8 -> 100 trên 16 mẫu (5 ca lật, 0 ca xấu đi) vẫn chưa đạt p < 0.05.

    Test này giữ cho framework trung thực: cải thiện nhìn rất to vẫn có thể
    không phân biệt được với ngẫu nhiên khi bộ mẫu quá nhỏ.
    """
    assert mcnemar_exact(0, 5) == pytest.approx(0.0625, abs=1e-4)
    assert mcnemar_exact(0, 5) >= 0.05


def test_mcnemar_can_6_ca_lat_moi_du():
    assert mcnemar_exact(0, 6) == pytest.approx(0.03125, abs=1e-5)
    assert min_flips_for_significance(0.05) == 6


def test_mcnemar_khong_bat_dong_thi_khong_co_bang_chung():
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_doi_xung():
    assert mcnemar_exact(3, 7) == mcnemar_exact(7, 3)


def test_mcnemar_bat_dong_can_bang_thi_khong_co_khac_biet():
    assert mcnemar_exact(5, 5) == 1.0


# ---------- ca không chấm được không được tính là bất đồng --------------
def test_discordant_bo_qua_ca_loi():
    """Ca lỗi (None) phải bị bỏ, nếu không lỗi mạng sẽ bị đếm thành khác biệt."""
    a = [True, False, None, True]
    b = [True, True, True, None]
    only_a, only_b = discordant_counts(a, b)
    assert (only_a, only_b) == (0, 1)   # chỉ cặp thứ 2 hợp lệ và bất đồng


def test_discordant_khac_do_dai_thi_loi():
    with pytest.raises(ValueError):
        discordant_counts([True], [True, False])


# ---------- non-inferiority ---------------------------------------------
def test_non_inferiority_bo_mau_nho_khong_ket_luan_duoc():
    """Điểm y hệt nhau nhưng n=16 vẫn KHÔNG chứng minh được 'không tệ hơn'.

    Đây là cạm bẫy chính: p > 0.05 không có nghĩa là bằng nhau.
    """
    assert non_inferiority(15, 15, 16, margin_pp=5.0) is False


def test_non_inferiority_du_mau_thi_ket_luan_duoc():
    assert non_inferiority(180, 180, 200, margin_pp=5.0) is True


def test_non_inferiority_bat_prompt_te_di():
    # rơi từ 90% xuống 70% thì không thể coi là 'không tệ hơn 5 điểm'
    assert non_inferiority(180, 140, 200, margin_pp=5.0) is False


# ---------- EvalResult mang được độ bất định ----------------------------
def _kq(dung: int, tong: int) -> EvalResult:
    samples = [Sample(id=i, text=f"t{i}", label="Yes") for i in range(tong)]
    preds = [Prediction(sample_id=i, output="Yes" if i < dung else "No")
             for i in range(tong)]
    return AccuracyEvaluator().evaluate("p", preds, samples)


def test_evalresult_co_khoang_tin_cay():
    r = _kq(16, 16)
    assert r.score == 100.0
    lo, hi = r.confidence_interval
    assert lo == pytest.approx(80.6, abs=0.1)
    assert r.margin_of_error > 9      # 100 điểm nhưng sai số ~±10


def test_100_va_93_8_khong_phan_biet_duoc_tren_16_mau():
    """Chính là chỗ mà score trần nói dối: hai điểm này chỉ hơn nhau 1 mẫu."""
    a = _kq(16, 16)
    b = _kq(15, 16)
    assert a.score - b.score == pytest.approx(6.2, abs=0.1)
    assert a.distinguishable_from(b) is False


def test_khac_biet_du_lon_thi_phan_biet_duoc():
    a = _kq(16, 16)
    b = _kq(8, 16)          # 8 ca lật -> p = 0.0078
    assert a.distinguishable_from(b) is True


# ---------- đo độ dài ----------------------------------------------------
def test_count_words():
    assert count_words("một hai ba") == 3
    assert count_words("  nhiều   khoảng   trắng  ") == 3
    assert count_words("") == 0


def test_composite_khong_phat_prompt_ngan():
    ev = CompositeEvaluator(word_budget=50, brevity_weight=10)
    samples = [Sample(id=0, text="t", label="Yes")]
    preds = [Prediction(sample_id=0, output="Yes")]
    r = ev.evaluate("ngắn gọn", preds, samples)
    assert r.score == 100.0
    assert r.metrics["brevity_penalty"] == 0.0


def test_composite_phat_prompt_dai():
    ev = CompositeEvaluator(word_budget=10, brevity_weight=10)
    samples = [Sample(id=0, text="t", label="Yes")]
    preds = [Prediction(sample_id=0, output="Yes")]
    prompt = " ".join(["từ"] * 20)          # gấp đôi ngân sách -> vượt 100%
    r = ev.evaluate(prompt, preds, samples)
    assert r.metrics["prompt_words"] == 20
    assert r.metrics["brevity_penalty"] == pytest.approx(10.0)
    assert r.score == 90.0
    assert r.metrics["accuracy"] == 100.0   # accuracy gốc vẫn giữ nguyên để báo cáo


def test_composite_khong_thuong_cho_ngan_hon_ngan_sach():
    """Ngắn hơn ngân sách KHÔNG được cộng điểm.

    Nếu thưởng, optimizer sẽ cắt prompt tới mức cụt lủn để ăn điểm ngắn.
    """
    ev = CompositeEvaluator(word_budget=100, brevity_weight=10)
    samples = [Sample(id=0, text="t", label="Yes")]
    preds = [Prediction(sample_id=0, output="Yes")]
    assert ev.evaluate("a", preds, samples).score == 100.0
    assert ev.evaluate(" ".join(["x"] * 99), preds, samples).score == 100.0


def test_composite_tham_so_sai():
    with pytest.raises(ValueError):
        CompositeEvaluator(word_budget=0)
    with pytest.raises(ValueError):
        CompositeEvaluator(brevity_weight=-1)


# ---------- đa model -----------------------------------------------------
def _preds_da_model(dung_theo_model):
    """dung_theo_model: {ten_model: so_ca_dung} trên 4 ca."""
    out = []
    for model, dung in dung_theo_model.items():
        for i in range(4):
            out.append(Prediction(sample_id=i, model=model,
                                  output="Yes" if i < dung else "No"))
    return out


_SAMPLES4 = [Sample(id=i, text=f"t{i}", label="Yes") for i in range(4)]


def test_cross_model_lay_model_te_nhat_khong_lay_trung_binh():
    """Model giỏi KHÔNG được che lấp model dở.

    A đúng 4/4 (100), B đúng 2/4 (50). Trung bình là 75 — nghe ổn nhưng đây
    KHÔNG phải prompt dùng được cho nhiều model. Điểm phải là 50.
    """
    ev = CrossModelEvaluator()
    r = ev.evaluate("p", _preds_da_model({"A": 4, "B": 2}), _SAMPLES4)
    assert r.score == 50.0
    assert r.metrics["accuracy_min"] == 50.0
    assert r.metrics["accuracy_mean"] == 75.0
    assert r.metadata["worst_model"] == "B"


def test_cross_model_bao_cao_do_chenh_lech():
    ev = CrossModelEvaluator()
    r = ev.evaluate("p", _preds_da_model({"A": 4, "B": 2}), _SAMPLES4)
    assert r.metrics["accuracy_spread"] == 50.0
    assert r.metrics["accuracy__A"] == 100.0
    assert r.metrics["accuracy__B"] == 50.0


def test_cross_model_loi_tra_ve_la_cua_model_te_nhat():
    """Optimizer phải nhận lỗi của model đang hỏng, chứ không phải model đang tốt."""
    ev = CrossModelEvaluator()
    r = ev.evaluate("p", _preds_da_model({"A": 4, "B": 2}), _SAMPLES4)
    assert len(r.errors) == 2
    assert all(e.model == "B" for e in r.errors)


def test_cross_model_khoang_tin_cay_khong_bi_hep_gia_tao():
    """Mẫu số phải là số MẪU, không phải số model × số mẫu.

    Gộp 2 model × 4 mẫu thành n=8 sẽ làm khoảng tin cậy hẹp đi giả tạo — đó là
    cùng 4 mẫu đo lặp lại, không phải 8 quan sát độc lập.
    """
    ev = CrossModelEvaluator()
    r = ev.evaluate("p", _preds_da_model({"A": 4, "B": 4}), _SAMPLES4)
    assert r.num_scored == 4, "không được cộng dồn mẫu của các model"
    assert r.confidence_interval == wilson_interval(4, 4)


def test_cross_model_mot_model_thi_hanh_xu_nhu_thuong():
    ev = CrossModelEvaluator()
    r = ev.evaluate("p", _preds_da_model({"A": 3}), _SAMPLES4)
    assert r.score == 75.0


def test_cross_model_mot_model_khong_dang_tin_thi_ca_ket_qua_khong_dang_tin():
    """Model nào chấm không đủ ca thì cả kết quả phải bị đánh dấu không đáng tin.

    Nếu không, model đó có thể 'được' điểm cao chỉ vì phần lớn ca của nó lỗi và
    bị loại khỏi mẫu số — đúng bẫy thổi phồng điểm.
    """
    ev = CrossModelEvaluator(min_scored_ratio=0.8)
    preds = [Prediction(sample_id=i, model="A", output="Yes") for i in range(4)]
    # model B: 3/4 ca lỗi -> chỉ chấm được 1 ca, và ca đó đúng -> 100 điểm giả
    preds.append(Prediction(sample_id=0, model="B", output="Yes"))
    preds += [Prediction(sample_id=i, model="B", output="__ERROR__: 429")
              for i in range(1, 4)]
    r = ev.evaluate("p", preds, _SAMPLES4)
    assert r.reliable is False
