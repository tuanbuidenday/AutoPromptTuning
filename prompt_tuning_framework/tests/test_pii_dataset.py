"""Canh giữ chất lượng bộ mẫu PII (lộ thông tin khách hàng).

Cùng một nguyên tắc với test_dataset.py: benchmark chỉ có giá trị khi LUẬT LƯỜI
không ăn được điểm cao. Bài này nguy hiểm hơn bộ ticket ở một điểm — prompt khởi
đầu có sẵn chữ "nhạy cảm", nên nếu bộ mẫu để chữ đó nghiêng về Yes thì model chỉ
cần học "thấy chữ nhạy cảm -> Yes" là xong, và ta đang đo TỪ KHOÁ chứ không đo
việc có lộ dữ liệu hay không. Prompt tối ưu thật sự đã tự viết ra rằng chữ đó vô
nghĩa; điều đó chỉ đáng tin nếu bộ mẫu thật sự cân.

Các test dưới khoá chặt sự cân bằng đó.
"""
import re
from pathlib import Path

import pytest

from prompt_tuning_framework import load_samples_csv, split_samples
from prompt_tuning_framework.examples.make_pii import N_TEST, SEED, _sinh

DATASET = Path(__file__).parent.parent / "examples" / "pii.csv"

TU_NHAY_CAM = re.compile(r"nhạy cảm|bảo mật", re.IGNORECASE)
CHUOI_SO = re.compile(r"\d{4,}")


@pytest.fixture(scope="module")
def mau():
    return load_samples_csv(str(DATASET))


def _ti_le_yes(mau, dieu_kien):
    co = [s for s in mau if dieu_kien(s.text)]
    assert co, "Không có ca nào khớp — điều kiện sai?"
    return sum(1 for s in co if s.label == "Yes") / len(co) * 100


def _diem_neu_doan_bang(mau, luat):
    dung = sum(1 for s in mau
               if (luat(s.text) and s.label == "Yes")
               or (not luat(s.text) and s.label == "No"))
    return dung / len(mau) * 100


# ---------- kích thước & cân bằng ---------------------------------------
def test_has_120_samples(mau):
    assert len(mau) == 120


def test_labels_are_balanced(mau):
    yes = sum(1 for s in mau if s.label == "Yes")
    assert yes == 60 and len(mau) - yes == 60


def test_no_duplicates(mau):
    texts = [s.text for s in mau]
    assert len(set(texts)) == len(texts), "Có ca trùng lặp"


# ---------- luật lười phải thất bại -------------------------------------
def test_sensitive_keyword_is_useless_for_guessing(mau):
    """P(Yes | có chữ "nhạy cảm"/"bảo mật") phải đúng 50%.

    Đây là test QUAN TRỌNG NHẤT của file. Prompt khởi đầu của bài toán chứa sẵn
    chữ "nhạy cảm"; nếu tỉ lệ này lệch khỏi 50 thì model bám vào từ khoá là đã ăn
    điểm, và kết quả 66.7 -> 100 không còn chứng minh nó hiểu khái niệm "lộ dữ
    liệu" nữa. Lệch mà không ai biết thì cả benchmark trở nên vô nghĩa một cách
    thầm lặng.
    """
    assert _ti_le_yes(mau, TU_NHAY_CAM.search) == 50.0


def test_keyword_rule_is_no_better_than_chance(mau):
    """Đoán bằng "có chữ nhạy cảm -> Yes" chỉ được đúng 50/100."""
    assert _diem_neu_doan_bang(mau, TU_NHAY_CAM.search) == 50.0


def test_digits_alone_are_not_enough(mau):
    """Ca No cũng phải có chuỗi số (mã đơn, hotline, tiền, ngày).

    Nếu chỉ ca Yes mới có số thì "thấy số -> Yes" là luật thắng, và benchmark
    đang đo sự hiện diện của chữ số chứ không đo định danh cá nhân.
    """
    assert _diem_neu_doan_bang(mau, CHUOI_SO.search) < 60


def test_customer_word_is_not_enough(mau):
    """Chữ "khách" xuất hiện ở cả hai nhãn — nêu tên khách không phải là lộ."""
    assert 40 <= _ti_le_yes(mau, lambda t: "khách" in t.lower()) <= 60


def test_no_single_lazy_rule_wins(mau):
    """Không luật bề mặt đơn lẻ nào vượt 60/100.

    Chốt chặn tổng: gom mọi tín hiệu bề mặt mà một model lười có thể bám vào.
    """
    luat = {
        "nhạy cảm": TU_NHAY_CAM.search,
        "chuỗi số": CHUOI_SO.search,
        "có @": lambda t: "@" in t,
        "có chữ khách": lambda t: "khách" in t.lower(),
    }
    qua = {ten: _diem_neu_doan_bang(mau, f) for ten, f in luat.items()
           if _diem_neu_doan_bang(mau, f) > 60}
    assert not qua, f"Luật lười ăn được điểm cao: {qua}"


# ---------- generator ----------------------------------------------------
def test_generator_is_deterministic():
    """Seed cố định -> sinh lại y hệt, nếu không thì kết quả không tái tạo được."""
    import random
    a = _sinh(random.Random(SEED))
    b = _sinh(random.Random(SEED))
    assert a == b


def test_generator_matches_written_file(mau):
    """pii.csv phải khớp generator — file cũ mà code mới thì số liệu là giả."""
    import random
    sinh = {t for t, _ in _sinh(random.Random(SEED))}
    assert {s.text for s in mau} == sinh


# ---------- tách train/test ---------------------------------------------
def test_split_does_not_leak(mau):
    dev, test = split_samples(mau, test_size=N_TEST, seed=0)
    assert len(test) == N_TEST
    assert not ({s.text for s in dev} & {s.text for s in test}), "Rò rỉ train/test"


def test_test_split_stays_balanced(mau):
    _, test = split_samples(mau, test_size=N_TEST, seed=0)
    yes = sum(1 for s in test if s.label == "Yes")
    assert abs(yes - len(test) / 2) <= 1, f"Tập test lệch nhãn: {yes}/{len(test)}"
