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
def mau():
    return load_samples_csv(str(DATASET))


def _khop(text: str, pool) -> bool:
    low = text.lower()
    return any(x.lower() in low for x in pool)


# ---------- kích thước & cân bằng ---------------------------------------
def test_du_400_mau(mau):
    assert len(mau) == 400


def test_can_bang_nhan(mau):
    """Lệch nhãn thì prompt ngu ngốc luôn trả nhãn đa số đã được điểm cao.

    25/75 -> prompt luôn trả 'No' được 75 điểm mà không hiểu gì.
    """
    n_yes = sum(1 for s in mau if s.label == "Yes")
    assert n_yes == 200
    assert len(mau) - n_yes == 200


def test_khong_trung_lap(mau):
    assert len({s.text for s in mau}) == len(mau)


# ---------- luật lười phải THẤT BẠI -------------------------------------
def _diem_luat(mau, luat) -> float:
    dung = sum(1 for s in mau
               if (luat(s.text) and s.label == "Yes")
               or (not luat(s.text) and s.label == "No"))
    return dung / len(mau) * 100


def test_giong_dieu_vo_dung_de_doan(mau):
    """Đây là test quan trọng nhất của file.

    Nếu chỉ ticket No mới gào to thì model chỉ cần học 'gào to -> No', và ta
    đang đo giọng điệu thay vì đo quy định nội bộ.
    """
    n_gao = sum(1 for s in mau if GAO.search(s.text))
    gao_yes = sum(1 for s in mau if GAO.search(s.text) and s.label == "Yes")
    ty_le = gao_yes / n_gao * 100
    assert 45 <= ty_le <= 55, (
        f"P(Yes | gào to) = {ty_le:.1f}% — giọng điệu đang dự đoán được nhãn. "
        f"Ticket gào to phải rải đều cả Yes lẫn No.")


def test_luat_theo_giong_dieu_chi_bang_doan_mo(mau):
    assert _diem_luat(mau, lambda t: bool(GAO.search(t))) < 56
    assert _diem_luat(mau, lambda t: not GAO.search(t)) < 56


def test_chi_dau_hieu_tra_tien_thi_chua_du(mau):
    """'Khách trả tiền -> Yes' phải KHÔNG đủ: khách trả tiền hỏi han vẫn là No."""
    diem = _diem_luat(mau, lambda t: _khop(t, PAYING))
    assert diem < 85, f"'trả tiền -> Yes' đạt {diem:.1f} — quá cao, thiếu ca khách trả tiền mà không bị chặn"


def test_chi_dau_hieu_bi_chan_thi_chua_du(mau):
    """'Bị chặn -> Yes' phải KHÔNG đủ: user free bị chặn vẫn là No.

    Đây là các ca then chốt. Không có chúng thì 'outage = Yes' ăn trọn benchmark
    và chiều TRẢ TIỀN không hề được kiểm tra.
    """
    diem = _diem_luat(mau, lambda t: _khop(t, BLOCKED))
    assert diem < 85, f"'bị chặn -> Yes' đạt {diem:.1f} — quá cao, thiếu ca free bị chặn"


# ---------- nhãn phải khớp thiết kế -------------------------------------
def test_nhan_khop_thiet_ke(mau):
    """Yes <=> trả tiền VÀ bị chặn. Không ngoại lệ."""
    sai = []
    for s in mau:
        p, f = _khop(s.text, PAYING), _khop(s.text, FREE)
        b, nb = _khop(s.text, BLOCKED), _khop(s.text, NOT_BLOCKED)
        mong_doi = "Yes" if (p and not f and b and not nb) else "No"
        if mong_doi != s.label:
            sai.append((s.text[:70], s.label, mong_doi))
    assert not sai, f"{len(sai)} ca nhãn sai thiết kế, vd: {sai[:3]}"


def test_moi_o_trong_thiet_ke_deu_co_mau(mau):
    """Đủ 4 ô: trả-tiền×chặn, trả-tiền×không-chặn, free×chặn, free×không-chặn."""
    o = {"tra_tien+chan": 0, "tra_tien+khong_chan": 0,
         "free+chan": 0, "free+khong_chan": 0}
    for s in mau:
        p = _khop(s.text, PAYING)
        b = _khop(s.text, BLOCKED)
        key = f"{'tra_tien' if p else 'free'}+{'chan' if b else 'khong_chan'}"
        o[key] += 1
    for ten, n in o.items():
        assert n >= 30, f"ô {ten} chỉ có {n} ca — quá ít để đo"
    # Ca then chốt: bị chặn hoàn toàn nhưng KHÔNG trả tiền -> vẫn No
    assert o["free+chan"] >= 60


# ---------- generator ----------------------------------------------------
def test_generator_tat_dinh():
    """Cùng seed -> cùng bộ mẫu, để mọi lần chạy so sánh được với nhau."""
    assert build() == build()


def test_generator_khop_file_da_ghi(mau):
    """File CSV phải đúng là thứ generator sinh ra — tránh sửa tay rồi quên."""
    rows = build()
    assert len(rows) == len(mau)
    assert {t for t, _ in rows} == {s.text for s in mau}
