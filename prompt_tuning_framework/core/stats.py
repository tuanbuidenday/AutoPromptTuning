"""Thống kê cho việc đo hiệu quả prompt.

Vì sao cần: điểm accuracy trần là một con số gây hiểu nhầm. Với 16 mẫu, 1 mẫu
= 6.25 điểm, nên "100.0" và "93.8" chỉ hơn nhau ĐÚNG MỘT MẪU — không thể kết
luận prompt nào tốt hơn. Muốn nói "prompt ngắn hơn mà accuracy không đổi" thì
bắt buộc phải đo được độ bất định, nếu không ta chỉ đang đọc nhiễu.

Không phụ thuộc scipy/numpy: chỉ dùng math + itertools của thư viện chuẩn.
"""
from functools import lru_cache
from math import comb, sqrt
from typing import List, Optional, Tuple

# z ứng với mức tin cậy hai phía 95%
Z_95 = 1.959963984540054


def wilson_interval(num_correct: int, num_total: int,
                    z: float = Z_95) -> Tuple[float, float]:
    """Khoảng tin cậy Wilson cho tỉ lệ đúng, trả về (thấp, cao) thang 0..100.

    Dùng Wilson thay vì công thức Wald (p ± z·√(p(1-p)/n)) vì Wald vỡ hoàn toàn
    ở rìa: với 16/16 đúng, Wald cho ra [100, 100] — tuyên bố chắc chắn tuyệt đối
    từ 16 mẫu. Wilson cho [80.6, 100], phản ánh đúng thực tế là 16 mẫu không đủ
    để loại trừ khả năng prompt chỉ đúng ~81%.

    :param num_correct: số ca đúng
    :param num_total: số ca được chấm (KHÔNG tính ca lỗi/không chấm được)
    """
    if num_total <= 0:
        return (0.0, 0.0)
    if not 0 <= num_correct <= num_total:
        raise ValueError(f"num_correct={num_correct} phải nằm trong [0, {num_total}]")

    z2 = z * z
    denom = num_total + z2
    center = (num_correct + z2 / 2) / denom
    half = z / denom * sqrt(num_correct * (num_total - num_correct) / num_total + z2 / 4)
    return (max(0.0, center - half) * 100, min(1.0, center + half) * 100)


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) với X ~ Nhị thức(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k + 1))


# Cache vì hàm này thuần tuý (cùng input -> cùng output) nhưng đắt: mỗi lần gọi
# là ~60 vòng dò nhị phân, mỗi vòng tính CDF nhị thức O(n). Số cặp (đúng, tổng)
# gặp thực tế rất ít — cùng một bộ test được chấm lại nhiều lần — nên cache ăn
# gần như tuyệt đối. Không cache thì test bao phủ mất 164s, có cache còn ~1s.
@lru_cache(maxsize=4096)
def clopper_pearson_interval(num_correct: int, num_total: int,
                             alpha: float = 0.05) -> Tuple[float, float]:
    """Khoảng tin cậy Clopper-Pearson ("exact"), trả về (thấp, cao) thang 0..100.

    Khi nào dùng thay Wilson: khi accuracy sát 0% hoặc 100%. Bao phủ của Wilson
    DAO ĐỘNG và tụt xuống ~92% ở vùng biên (đặc tính đã biết của phương pháp — đã
    kiểm chứng bằng mô phỏng rằng statsmodels cũng tụt y hệt, xem
    tests/test_do_luong_dung_khong.py). Clopper-Pearson bảo đảm bao phủ >= 95% ở
    MỌI p, đổi lại khoảng rộng hơn một chút.

    Mặc định của framework vẫn là Wilson vì nó chặt hơn và là lựa chọn tiêu chuẩn;
    dùng hàm này khi cần bảo đảm chắc chắn ở vùng biên.

    Cài đặt bằng cách dò nhị phân trên hàm phân phối nhị thức — tương đương với
    phân vị Beta nhưng không cần scipy:
        cận dưới = p sao cho P(X >= x | p) = alpha/2
        cận trên = p sao cho P(X <= x | p) = alpha/2
    """
    if num_total <= 0:
        return (0.0, 0.0)
    if not 0 <= num_correct <= num_total:
        raise ValueError(f"num_correct={num_correct} phải nằm trong [0, {num_total}]")

    def _do(muc_tieu, la_can_duoi):
        lo, hi = 0.0, 1.0
        # 60 vòng -> sai số ~1e-18, thừa xa so với độ chính xác cần thiết
        # (test đối chiếu statsmodels dùng dung sai 1e-6).
        for _ in range(60):
            mid = (lo + hi) / 2
            if la_can_duoi:
                # P(X >= x | p) tăng theo p
                val = 1 - _binom_cdf(num_correct - 1, num_total, mid)
                lo, hi = (lo, mid) if val > muc_tieu else (mid, hi)
            else:
                # P(X <= x | p) giảm theo p
                val = _binom_cdf(num_correct, num_total, mid)
                lo, hi = (mid, hi) if val > muc_tieu else (lo, mid)
        return (lo + hi) / 2

    thap = 0.0 if num_correct == 0 else _do(alpha / 2, True)
    cao = 1.0 if num_correct == num_total else _do(alpha / 2, False)
    return (thap * 100, cao * 100)


def mcnemar_exact(only_a_correct: int, only_b_correct: int) -> float:
    """Kiểm định McNemar (bản exact) so hai prompt trên CÙNG bộ mẫu.

    Dùng kiểm định ghép cặp chứ không so hai khoảng tin cậy rời, vì hai prompt
    chạy trên cùng các ca: những ca cả hai cùng đúng (hoặc cùng sai) không mang
    thông tin phân biệt. Chỉ các ca BẤT ĐỒNG mới đáng kể, và tận dụng điều đó
    cho lực kiểm định mạnh hơn nhiều so với việc so hai khoảng tin cậy độc lập.

    Bản exact (nhị thức) chứ không phải xấp xỉ chi-bình-phương, vì số ca bất
    đồng ở đây thường rất nhỏ (dưới 25) — chỗ mà xấp xỉ không còn đúng.

    :param only_a_correct: số ca A đúng mà B sai
    :param only_b_correct: số ca A sai mà B đúng
    :return: p-value hai phía. Nhỏ = khác biệt khó xảy ra do ngẫu nhiên.
    """
    if only_a_correct < 0 or only_b_correct < 0:
        raise ValueError("Số ca bất đồng không thể âm.")

    n = only_a_correct + only_b_correct
    if n == 0:
        return 1.0  # không có ca nào bất đồng -> không có bằng chứng khác biệt

    k = min(only_a_correct, only_b_correct)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def min_flips_for_significance(alpha: float = 0.05, max_n: int = 100) -> Optional[int]:
    """Số ca ít nhất phải lật từ SAI sang ĐÚNG (và không ca nào xấu đi) để đạt
    mức ý nghĩa alpha.

    Có hàm này để trả lời trước khi chạy: "bộ mẫu của tôi có đủ để chứng minh
    điều gì không?". Với alpha=0.05 câu trả lời là 6 — nghĩa là một cải thiện
    lật được 5 ca vẫn CHƯA đủ, dù nhìn rất ấn tượng.
    """
    for c in range(1, max_n + 1):
        if mcnemar_exact(0, c) < alpha:
            return c
    return None


def non_inferiority(base_correct: int, new_correct: int, num_total: int,
                    margin_pp: float, z: float = Z_95) -> bool:
    """Prompt mới có 'không tệ hơn quá margin_pp điểm' so với prompt gốc không?

    Đây là câu hỏi ĐÚNG khi rút gọn prompt: ta muốn chứng minh accuracy KHÔNG
    tụt, chứ không phải chứng minh nó tăng.

    Cạm bẫy phải tránh: không được suy "p > 0.05 nên hai prompt bằng nhau".
    Không bác bỏ được H0 không có nghĩa là H0 đúng — với bộ mẫu nhỏ thì p LUÔN
    lớn hơn 0.05, nên lập luận đó sẽ luôn kết luận "không đổi" kể cả khi prompt
    mới tệ đi thật. Cách đúng là đảo ngược gánh nặng chứng minh: chỉ chấp nhận
    khi CẬN DƯỚI của khoảng tin cậy cho prompt mới vẫn nằm trên (accuracy gốc
    trừ margin).

    :param margin_pp: ngưỡng chấp nhận, tính bằng điểm phần trăm (vd 5.0)
    :return: True nếu có bằng chứng prompt mới không tệ hơn quá ngưỡng
    """
    if num_total <= 0:
        return False
    if margin_pp < 0:
        raise ValueError("margin_pp không được âm.")

    base_rate = base_correct / num_total * 100
    new_lo, _ = wilson_interval(new_correct, num_total, z=z)
    return new_lo >= base_rate - margin_pp


def discordant_counts(correct_a: List[Optional[bool]],
                      correct_b: List[Optional[bool]]) -> Tuple[int, int]:
    """Đếm ca bất đồng giữa hai lần chạy trên cùng bộ mẫu.

    Ca nào có bên None (không chấm được vì LLM lỗi/quota) thì BỎ QUA cả cặp —
    giữ lại sẽ tính nhầm một ca hỏng thành một ca bất đồng, tức là lấy lỗi mạng
    làm bằng chứng prompt khác nhau.

    :return: (số ca chỉ A đúng, số ca chỉ B đúng)
    """
    if len(correct_a) != len(correct_b):
        raise ValueError("Hai danh sách phải cùng độ dài (cùng bộ mẫu).")

    only_a = only_b = 0
    for a, b in zip(correct_a, correct_b):
        if a is None or b is None:
            continue
        if a and not b:
            only_a += 1
        elif b and not a:
            only_b += 1
    return (only_a, only_b)
