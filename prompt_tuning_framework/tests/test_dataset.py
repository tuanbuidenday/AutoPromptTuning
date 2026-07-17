"""Canh giữ chất lượng bộ mẫu ticket.

Một benchmark chỉ có giá trị khi các LUẬT LƯỜI không ăn được điểm cao. Bộ 96 mẫu
viết tay trước đây hỏng đúng chỗ này: 15/15 ticket gào "URGENT!!!" đều là No, nên
"gào to -> No" là luật đúng 100% trong nhóm đó — benchmark đang đo giọng điệu chứ
không đo quy định thật.

Các test này khoá chặt thiết kế giai thừa để lỗi đó không quay lại.
"""
import re
from pathlib import Path

import pytest

from prompt_tuning_framework import load_samples_csv
from prompt_tuning_framework.examples.make_tickets import (BLOCKED, FREE,
                                                           NOT_BLOCKED, PAYING,
                                                           build)

DATASET = Path(__file__).parent.parent / "examples" / "tickets.csv"

GAO = re.compile(r"!!!|URGENT|CRITICAL|EMERGENCY|ASAP|PRIORITY|DISASTER|SEV1|"
                 r"BLOCKER|ESCALATE|INCIDENT")


@pytest.fixture(scope="module")
def tickets():
    return load_samples_csv(str(DATASET))


def _matches(text: str, pool) -> bool:
    low = text.lower()
    return any(x.lower() in low for x in pool)


# ---------- kích thước & cân bằng ---------------------------------------
def test_has_480_samples(tickets):
    """480 = 280 train + 200 test.

    200 ca test là cỡ nhỏ nhất còn kết luận được "prompt mới không tệ hơn 5
    điểm"; 120 ca chỉ đủ cho mức 7 điểm.
    """
    assert len(tickets) == 480


def test_labels_are_balanced(tickets):
    """Lệch nhãn thì prompt ngu ngốc luôn trả nhãn đa số đã được điểm cao.

    25/75 -> prompt luôn trả 'No' được 75 điểm mà không hiểu gì.
    """
    n_yes = sum(1 for s in tickets if s.label == "Yes")
    assert n_yes == 240
    assert len(tickets) - n_yes == 240


def test_splits_into_280_and_200(tickets):
    """Cách chia dùng trong hard_example phải ra đúng 280/200, cân bằng nhãn."""
    from prompt_tuning_framework import non_inferiority, split_samples

    dev, test = split_samples(tickets, test_size=200, seed=0)
    assert len(dev) == 280 and len(test) == 200
    assert sum(1 for s in test if s.label == "Yes") == 100
    assert sum(1 for s in dev if s.label == "Yes") == 140
    assert not ({s.text for s in dev} & {s.text for s in test})
    # Lý do tồn tại của con số 200: dưới mức này thì mục tiêu không kết luận nổi.
    assert non_inferiority(180, 180, len(test), margin_pp=5.0) is True


def test_no_duplicates(tickets):
    assert len({s.text for s in tickets}) == len(tickets)


# ---------- luật lười phải THẤT BẠI -------------------------------------
def _rule_score(tickets, luat) -> float:
    dung = sum(1 for s in tickets
               if (luat(s.text) and s.label == "Yes")
               or (not luat(s.text) and s.label == "No"))
    return dung / len(tickets) * 100


def test_tone_is_useless_for_guessing(tickets):
    """Đây là test quan trọng nhất của file.

    Nếu chỉ ticket No mới gào to thì model chỉ cần học 'gào to -> No', và ta
    đang đo giọng điệu thay vì đo quy định nội bộ.
    """
    n_gao = sum(1 for s in tickets if GAO.search(s.text))
    gao_yes = sum(1 for s in tickets if GAO.search(s.text) and s.label == "Yes")
    ty_le = gao_yes / n_gao * 100
    assert 45 <= ty_le <= 55, (
        f"P(Yes | gào to) = {ty_le:.1f}% — giọng điệu đang dự đoán được nhãn. "
        f"Ticket gào to phải rải đều cả Yes lẫn No.")


def test_tone_rule_is_no_better_than_chance(tickets):
    assert _rule_score(tickets, lambda t: bool(GAO.search(t))) < 56
    assert _rule_score(tickets, lambda t: not GAO.search(t)) < 56


def test_paying_signal_alone_is_not_enough(tickets):
    """'Khách trả tiền -> Yes' phải KHÔNG đủ: khách trả tiền hỏi han vẫn là No."""
    diem = _rule_score(tickets, lambda t: _matches(t, PAYING))
    assert diem < 85, f"'trả tiền -> Yes' đạt {diem:.1f} — quá cao, thiếu ca khách trả tiền mà không bị chặn"


def test_blocked_signal_alone_is_not_enough(tickets):
    """'Bị chặn -> Yes' phải KHÔNG đủ: user free bị chặn vẫn là No.

    Đây là các ca then chốt. Không có chúng thì 'outage = Yes' ăn trọn benchmark
    và chiều TRẢ TIỀN không hề được kiểm tra.
    """
    diem = _rule_score(tickets, lambda t: _matches(t, BLOCKED))
    assert diem < 85, f"'bị chặn -> Yes' đạt {diem:.1f} — quá cao, thiếu ca free bị chặn"


# ---------- nhãn phải khớp thiết kế -------------------------------------
def test_labels_match_the_design(tickets):
    """Yes <=> trả tiền VÀ bị chặn. Không ngoại lệ."""
    sai = []
    for s in tickets:
        p, f = _matches(s.text, PAYING), _matches(s.text, FREE)
        b, nb = _matches(s.text, BLOCKED), _matches(s.text, NOT_BLOCKED)
        mong_doi = "Yes" if (p and not f and b and not nb) else "No"
        if mong_doi != s.label:
            sai.append((s.text[:70], s.label, mong_doi))
    assert not sai, f"{len(sai)} ca nhãn sai thiết kế, vd: {sai[:3]}"


def test_every_design_cell_has_samples(tickets):
    """Đủ 4 ô: trả-tiền×chặn, trả-tiền×không-chặn, free×chặn, free×không-chặn."""
    o = {"tra_tien+chan": 0, "tra_tien+khong_chan": 0,
         "free+chan": 0, "free+khong_chan": 0}
    for s in tickets:
        p = _matches(s.text, PAYING)
        b = _matches(s.text, BLOCKED)
        key = f"{'tra_tien' if p else 'free'}+{'chan' if b else 'khong_chan'}"
        o[key] += 1
    for ten, n in o.items():
        assert n >= 30, f"ô {ten} chỉ có {n} ca — quá ít để đo"
    # Ca then chốt: bị chặn hoàn toàn nhưng KHÔNG trả tiền -> vẫn No
    assert o["free+chan"] >= 60


# ---------- generator ----------------------------------------------------
def test_generator_is_deterministic():
    """Cùng seed -> cùng bộ mẫu, để mọi lần chạy so sánh được với nhau."""
    assert build() == build()


def test_generator_matches_written_file(tickets):
    """File CSV phải đúng là thứ generator sinh ra — tránh sửa tay rồi quên."""
    rows = build()
    assert len(rows) == len(tickets)
    assert {t for t, _ in rows} == {s.text for s in tickets}


# ---------- file train/test xuất ra phải luôn khớp -----------------------
F_TRAIN = Path(__file__).parent.parent / "examples" / "tickets_train.csv"
F_TEST = Path(__file__).parent.parent / "examples" / "tickets_test.csv"


def test_train_test_files_match_split_samples(tickets):
    """Hai file xuất ra phải khớp CHÍNH XÁC thứ split_samples sinh ra lúc chạy.

    Đây là test quan trọng nhất của cặp file này. Chúng chỉ để người đọc mở ra
    xem — code thật vẫn gọi split_samples. Nếu bộ mẫu đổi mà quên chạy lại
    make_tickets, hai file sẽ âm thầm thành DỮ LIỆU MA: trông đúng, mở ra đọc
    được, nhưng không phải tập test thật đã dùng để ra con số trong báo cáo.
    """
    from prompt_tuning_framework import split_samples
    from prompt_tuning_framework.examples.make_tickets import N_TEST, SPLIT_SEED

    dev, test = split_samples(tickets, test_size=N_TEST, seed=SPLIT_SEED)
    f_train = load_samples_csv(str(F_TRAIN))
    f_test = load_samples_csv(str(F_TEST))

    assert [s.text for s in f_train] == [s.text for s in dev], \
        "tickets_train.csv lệch — chạy lại: python -m prompt_tuning_framework.examples.make_tickets"
    assert [s.text for s in f_test] == [s.text for s in test], \
        "tickets_test.csv lệch — chạy lại: python -m prompt_tuning_framework.examples.make_tickets"
    assert [s.label for s in f_test] == [s.label for s in test]


def test_train_test_files_do_not_leak():
    """Một ca nằm ở cả hai tập là hỏng toàn bộ ý nghĩa của tập test."""
    f_train = load_samples_csv(str(F_TRAIN))
    f_test = load_samples_csv(str(F_TEST))
    assert not ({s.text for s in f_train} & {s.text for s in f_test})
    assert len(f_train) == 280 and len(f_test) == 200


def test_test_file_labels_are_balanced():
    f_test = load_samples_csv(str(F_TEST))
    assert sum(1 for s in f_test if s.label == "Yes") == 100
