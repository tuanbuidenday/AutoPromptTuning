"""Chốt chặn tự động — thay cho việc grep tay.

Lý do có file này: một lệnh grep tay đã từng báo "sạch" trong khi thực ra nó
không hề chạy (shell nuốt mất tham số), suýt bỏ lọt chỗ ghim cứng tên model
trong UI. Kiểm tra tự động thì không nói dối được.
"""
import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent

# Chỉ llm.py được phép chứa tên model — đó là bảng DEFAULT_MODELS.
# examples/ được phép vì ví dụ cố ý chỉ ra model cụ thể.
FILE_DUOC_PHEP = {"llm.py"}
THU_MUC_BO_QUA = {"tests", "examples", "__pycache__"}

TU_KHOA_MODEL = ("gemini-", "gpt-3", "gpt-4", "gpt-5", "claude-")


def _file_py():
    for p in sorted(PKG.rglob("*.py")):
        if THU_MUC_BO_QUA & set(p.relative_to(PKG).parts):
            continue
        yield p


def _chuoi_trong_code(path: Path):
    """Các hằng chuỗi nằm trong CODE, bỏ qua docstring.

    Docstring nêu tên model là tài liệu, không phải giá trị mặc định -> không tính.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    id_docstring = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            if ast.get_docstring(node, clean=False) is not None:
                id_docstring.add(id(node.body[0].value))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in id_docstring):
            yield node.lineno, node.value


def test_khong_ghim_cung_ten_model_ngoai_llm_py():
    """Tên model chỉ được sống ở llm.py::DEFAULT_MODELS.

    Đây chính là bug đã xảy ra: LLMExecutor ghim 'gemini-2.5-flash-lite' nên
    --provider openai vẫn gửi tên model Gemini sang OpenAI.
    """
    pham = []
    for path in _file_py():
        if path.name in FILE_DUOC_PHEP:
            continue
        for lineno, value in _chuoi_trong_code(path):
            if any(tu in value.lower() for tu in TU_KHOA_MODEL):
                pham.append(f"{path.relative_to(PKG)}:{lineno} -> {value!r}")
    assert not pham, (
        "Tên model bị ghim cứng ngoài llm.py::DEFAULT_MODELS:\n  "
        + "\n  ".join(pham)
        + "\nHãy dùng default_model(provider, role) thay vì viết thẳng tên model."
    )


def test_guard_that_su_bat_duoc_loi(tmp_path):
    """Kiểm tra chính cái guard — guard hỏng mà im lặng thì vô dụng."""
    xau = tmp_path / "xau.py"
    xau.write_text('MODEL = "gemini-2.5-flash-lite"\n', encoding="utf-8")
    tim_thay = [v for _, v in _chuoi_trong_code(xau)
                if any(t in v.lower() for t in TU_KHOA_MODEL)]
    assert tim_thay == ["gemini-2.5-flash-lite"]


def test_guard_bo_qua_docstring(tmp_path):
    """Docstring nêu tên model là tài liệu hợp lệ -> guard không được báo nhầm."""
    tot = tmp_path / "tot.py"
    tot.write_text('"""Ví dụ: model gemini-2.5-flash-lite."""\nX = 1\n', encoding="utf-8")
    tim_thay = [v for _, v in _chuoi_trong_code(tot)
                if any(t in v.lower() for t in TU_KHOA_MODEL)]
    assert tim_thay == []


@pytest.mark.parametrize("ten", ["llm.py", "cli.py", "config.py", "data.py"])
def test_cac_file_loi_van_ton_tai(ten):
    """Đổi tên/xoá file lõi thì guard ở trên sẽ quét nhầm phạm vi mà không ai biết."""
    assert (PKG / ten).is_file(), f"Thiếu {ten} — guard quét sai phạm vi?"


def test_khong_co_khoa_api_that_trong_ma_nguon():
    """Không được có khoá API nào lọt vào mã nguồn của framework."""
    pham = []
    for path in _file_py():
        for lineno, value in _chuoi_trong_code(path):
            v = value.strip()
            if v.startswith("sk-") and len(v) > 20:
                pham.append(f"{path.relative_to(PKG)}:{lineno} (khoá OpenAI?)")
            if v.startswith("AQ.") and len(v) > 20:
                pham.append(f"{path.relative_to(PKG)}:{lineno} (khoá Google?)")
    assert not pham, "Nghi có khoá API trong mã nguồn:\n  " + "\n  ".join(pham)
