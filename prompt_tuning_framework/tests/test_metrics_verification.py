"""Kiểm chứng rằng CHÍNH CÁC PHÉP ĐO là đúng.

Câu hỏi mà file này trả lời: "làm sao chắc chắn phần đo lường không sai?"

Các file test khác kiểm tra framework dùng phép đo đúng cách. File này kiểm tra
bản thân phép đo — vì mọi kết luận trong báo cáo đều dựa lên nó, mà nó lại do
chính tác giả framework tự viết. Tự kiểm chứng công thức bằng chính công thức đó
là vô giá trị: hiểu sai từ đầu thì cả hai lần đều sai giống hệt nhau.

Ba tầng kiểm chứng, độc lập nhau:

1. ĐỐI CHIẾU NGOÀI — so với statsmodels, thư viện thống kê chuẩn do người khác
   viết, không biết gì về dự án này. Tự bỏ qua nếu statsmodels không có (CI chỉ
   cài extras [test]) — nên tầng 2 và 3 phải chạy được bằng stdlib thuần.

2. GIÁ TRỊ THAM CHIẾU — vài giá trị tính tay/tra sách, khoá cứng. Bắt được lỗi
   ngay cả khi statsmodels vắng mặt.

3. MÔ PHỎNG — kiểm tra TÍNH CHẤT THỐNG KÊ chứ không kiểm tra công thức. Khoảng
   tin cậy 95% phải chứa giá trị thật ~95% số lần; kiểm định mức 5% chỉ được báo
   động sai tối đa 5% số lần. Tầng này bắt được cả lỗi mà tầng 1 và 2 bỏ lọt —
   nếu tôi dùng SAI CÔNG CỤ (chứ không phải cài sai công thức), chỉ mô phỏng mới
   phát hiện.
"""
import random

import pytest

from prompt_tuning_framework.core.stats import (clopper_pearson_interval,
                                                mcnemar_exact, wilson_interval)

# ---------- tầng 1: đối chiếu với statsmodels ---------------------------
CA_TI_LE = [(16, 16), (15, 16), (11, 16), (0, 16), (180, 200), (200, 200),
            (0, 200), (1, 3), (50, 100), (99, 100), (0, 1), (1, 1), (7, 13)]


def test_wilson_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.stats.proportion")
    for x, n in CA_TI_LE:
        mine = wilson_interval(x, n)
        lo, hi = sm.proportion_confint(x, n, alpha=0.05, method="wilson")
        assert mine[0] == pytest.approx(lo * 100, abs=1e-6), f"{x}/{n}"
        assert mine[1] == pytest.approx(hi * 100, abs=1e-6), f"{x}/{n}"


def test_clopper_pearson_matches_statsmodels():
    """Bản tự viết dùng dò nhị phân trên CDF nhị thức; statsmodels dùng phân vị
    Beta của scipy. Hai đường tính hoàn toàn khác nhau mà ra cùng kết quả."""
    sm = pytest.importorskip("statsmodels.stats.proportion")
    for x, n in CA_TI_LE:
        mine = clopper_pearson_interval(x, n)
        lo, hi = sm.proportion_confint(x, n, alpha=0.05, method="beta")
        assert mine[0] == pytest.approx(lo * 100, abs=1e-6), f"{x}/{n}"
        assert mine[1] == pytest.approx(hi * 100, abs=1e-6), f"{x}/{n}"


def test_mcnemar_matches_statsmodels():
    ct = pytest.importorskip("statsmodels.stats.contingency_tables")
    np = pytest.importorskip("numpy")
    for b, c in [(0, 5), (0, 6), (0, 57), (3, 7), (5, 5), (0, 0), (1, 0),
                 (10, 2), (20, 30), (0, 1)]:
        mine = mcnemar_exact(b, c)
        theirs = ct.mcnemar(np.array([[100, b], [c, 100]]),
                            exact=True, correction=False).pvalue
        assert mine == pytest.approx(theirs, abs=1e-12), f"b={b}, c={c}"


# ---------- tầng 2: giá trị tham chiếu khoá cứng ------------------------
def test_wilson_reference_values():
    """Tính tay, khoá cứng — bắt lỗi ngay cả khi không có statsmodels."""
    for x, n, lo, hi in [(16, 16, 80.64, 100.00), (15, 16, 71.67, 98.89),
                         (11, 16, 44.40, 85.84), (180, 200, 85.06, 93.43),
                         (50, 100, 40.38, 59.62), (0, 16, 0.00, 19.36)]:
        a, b = wilson_interval(x, n)
        assert a == pytest.approx(lo, abs=0.01), f"{x}/{n} cận dưới"
        assert b == pytest.approx(hi, abs=0.01), f"{x}/{n} cận trên"


def test_mcnemar_reference_values():
    """b=0,c=k thì p = 2/2^k — suy được bằng tay, không cần thư viện."""
    for k in range(1, 12):
        assert mcnemar_exact(0, k) == pytest.approx(min(1.0, 2 / 2**k), rel=1e-12)


def test_the_real_reported_result():
    """Khoá cứng đúng hai con số đang được trích trong báo cáo."""
    # 57 ca lật sai->đúng, 0 ca xấu đi, trên tập test 200 ca
    assert mcnemar_exact(0, 57) == pytest.approx(1.3878e-17, rel=1e-3)
    lo, hi = wilson_interval(200, 200)
    assert (round(lo, 1), round(hi, 1)) == (98.1, 100.0)
    lo, hi = wilson_interval(143, 200)          # prompt gốc: 71.5/100
    assert round(lo, 1) == 64.9 and round(hi, 1) == 77.3


# ---------- tầng 3: mô phỏng — kiểm tra TÍNH CHẤT, không phải công thức --
def _coverage(p_that: float, n: int, ham, n_lan: int, seed: int) -> float:
    rng = random.Random(seed)
    trong = 0
    for _ in range(n_lan):
        x = sum(1 for _ in range(n) if rng.random() < p_that)
        lo, hi = ham(x, n)
        if lo <= p_that * 100 <= hi:
            trong += 1
    return trong / n_lan * 100


@pytest.mark.parametrize("p_that,n", [(0.5, 50), (0.9, 100), (0.9, 200), (0.7, 30)])
def test_wilson_coverage_is_really_95_percent(p_that, n):
    """Định nghĩa của "khoảng tin cậy 95%": nó phải chứa giá trị THẬT 95% số lần.

    Đây là kiểm chứng mạnh nhất — nó không dựa vào công thức nào cả, chỉ đo hành
    vi. Cài sai Wilson sẽ làm bao phủ lệch hẳn khỏi 95%.

    Ngưỡng nới [93, 97.5] vì bao phủ của Wilson dao động quanh mức danh nghĩa do
    phân phối nhị thức rời rạc — đây là đặc tính đã biết của phương pháp, không
    phải lỗi cài đặt (test_wilson_tut_bao_phu_o_bien chứng minh điều đó).
    """
    bp = _coverage(p_that, n, wilson_interval, n_lan=4000, seed=42)
    assert 93.0 <= bp <= 97.5, f"bao phủ {bp:.1f}% — lệch quá xa 95%"


def test_wilson_boundary_coverage_dip_is_a_property_not_a_bug():
    """Ở p sát 1, bao phủ Wilson tụt dưới 95%. Ghi lại để không ai "sửa" nhầm.

    Đã kiểm chứng bằng cách chạy CÙNG mô phỏng với statsmodels: nó tụt y hệt
    (92.2% vs 92.2%). Vậy đây là đặc tính của phương pháp Wilson (bao phủ dao
    động — Brown, Cai & DasGupta 2001), không phải lỗi cài đặt.

    Ảnh hưởng tới báo cáo: kết quả thật là 200/200 = 100%, tức p̂ ở ĐÚNG biên.
    May là ở p=1.0 bao phủ lại đạt 100%, và Wilson [98.1, 100] gần như trùng
    Clopper-Pearson [98.2, 100] — nên con số công bố vẫn an toàn.
    """
    bp = _coverage(0.99, 100, wilson_interval, n_lan=4000, seed=7)
    assert bp < 95.0, "nếu test này đỏ thì đặc tính của Wilson đã đổi — xem lại"


@pytest.mark.parametrize("p_that,n", [(0.99, 100), (0.995, 200), (0.9, 100)])
def test_clopper_pearson_never_drops_below_95(p_that, n):
    """Clopper-Pearson là "exact": bao đảm >= 95% ở MỌI p, kể cả sát biên.

    Đây là lý do nó tồn tại trong framework — chỗ Wilson yếu thì dùng nó.
    """
    bp = _coverage(p_that, n, clopper_pearson_interval, n_lan=2000, seed=11)
    assert bp >= 95.0, f"bao phủ {bp:.1f}% — Clopper-Pearson lẽ ra phải bảo thủ"


@pytest.mark.parametrize("p_that,n", [(0.9, 200), (0.8, 100), (0.5, 50)])
def test_mcnemar_false_alarm_stays_under_5_percent(p_that, n):
    """Khi hai prompt THỰC SỰ như nhau, kiểm định mức 5% chỉ được kết luận
    "khác nhau" tối đa 5% số lần.

    Nếu tỉ lệ này vượt 5%, framework sẽ tuyên bố cải thiện ở những chỗ chỉ có
    nhiễu — đúng thứ nó sinh ra để ngăn.
    """
    rng = random.Random(3)
    gia = 0
    n_lan = 2000
    for _ in range(n_lan):
        b = c = 0
        for _ in range(n):
            a_dung = rng.random() < p_that
            b_dung = rng.random() < p_that
            if a_dung and not b_dung:
                b += 1
            elif b_dung and not a_dung:
                c += 1
        if mcnemar_exact(b, c) < 0.05:
            gia += 1
    tl = gia / n_lan * 100
    assert tl <= 5.0, f"báo động giả {tl:.1f}% — vượt mức 5% đã tuyên bố"


def test_mcnemar_detects_a_real_difference():
    """Ngược lại: khi hai prompt KHÁC nhau thật, kiểm định phải nhận ra.

    Không có test này thì một hàm luôn trả p=1.0 vẫn qua được test báo-động-giả
    ở trên — bảo thủ tuyệt đối mà vô dụng hoàn toàn.
    """
    rng = random.Random(5)
    bat_duoc = 0
    n_lan = 500
    for _ in range(n_lan):
        b = c = 0
        for _ in range(200):
            a_dung = rng.random() < 0.70      # prompt A yếu hơn hẳn
            b_dung = rng.random() < 0.90      # prompt B
            if a_dung and not b_dung:
                b += 1
            elif b_dung and not a_dung:
                c += 1
        if mcnemar_exact(b, c) < 0.05:
            bat_duoc += 1
    tl = bat_duoc / n_lan * 100
    assert tl > 95.0, f"chỉ bắt được {tl:.1f}% khác biệt rõ rệt — lực kiểm định quá yếu"
