"""Chốt chặn tự động — thay cho việc grep tay.

Lý do có file này: một lệnh grep tay đã từng báo "sạch" trong khi thực ra nó
không hề chạy (shell nuốt mất tham số), suýt bỏ lọt chỗ ghim cứng tên model
trong UI. Kiểm tra tự động thì không nói dối được.
"""
import ast
import importlib.metadata
import re
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent

# Chỉ llm.py được phép chứa tên model — đó là bảng DEFAULT_MODELS.
# examples/ được phép vì ví dụ cố ý chỉ ra model cụ thể.
FILE_DUOC_PHEP = {"llm.py"}
THU_MUC_BO_QUA = {"tests", "examples", "__pycache__"}

TU_KHOA_MODEL = ("gemini-", "gpt-3", "gpt-4", "gpt-5", "claude-")


def _py_files():
    for p in sorted(PKG.rglob("*.py")):
        if THU_MUC_BO_QUA & set(p.relative_to(PKG).parts):
            continue
        yield p


def _strings_in_code(path: Path):
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


def test_no_hardcoded_model_names_outside_llm_py():
    """Tên model chỉ được sống ở llm.py::DEFAULT_MODELS.

    Đây chính là bug đã xảy ra: LLMExecutor ghim 'gemini-2.5-flash-lite' nên
    --provider openai vẫn gửi tên model Gemini sang OpenAI.
    """
    pham = []
    for path in _py_files():
        if path.name in FILE_DUOC_PHEP:
            continue
        for lineno, value in _strings_in_code(path):
            if any(tu in value.lower() for tu in TU_KHOA_MODEL):
                pham.append(f"{path.relative_to(PKG)}:{lineno} -> {value!r}")
    assert not pham, (
        "Tên model bị ghim cứng ngoài llm.py::DEFAULT_MODELS:\n  "
        + "\n  ".join(pham)
        + "\nHãy dùng default_model(provider, role) thay vì viết thẳng tên model."
    )


def test_guard_actually_catches_violations(tmp_path):
    """Kiểm tra chính cái guard — guard hỏng mà im lặng thì vô dụng."""
    xau = tmp_path / "xau.py"
    xau.write_text('MODEL = "gemini-2.5-flash-lite"\n', encoding="utf-8")
    tim_thay = [v for _, v in _strings_in_code(xau)
                if any(t in v.lower() for t in TU_KHOA_MODEL)]
    assert tim_thay == ["gemini-2.5-flash-lite"]


def test_guard_ignores_docstrings(tmp_path):
    """Docstring nêu tên model là tài liệu hợp lệ -> guard không được báo nhầm."""
    tot = tmp_path / "tot.py"
    tot.write_text('"""Ví dụ: model gemini-2.5-flash-lite."""\nX = 1\n', encoding="utf-8")
    tim_thay = [v for _, v in _strings_in_code(tot)
                if any(t in v.lower() for t in TU_KHOA_MODEL)]
    assert tim_thay == []


@pytest.mark.parametrize("ten", ["llm.py", "cli.py", "config.py", "data.py"])
def test_guarded_files_still_exist(ten):
    """Đổi tên/xoá file lõi thì guard ở trên sẽ quét nhầm phạm vi mà không ai biết."""
    assert (PKG / ten).is_file(), f"Thiếu {ten} — guard quét sai phạm vi?"


def test_no_real_api_keys_in_source():
    """Không được có khoá API nào lọt vào mã nguồn của framework."""
    pham = []
    for path in _py_files():
        for lineno, value in _strings_in_code(path):
            v = value.strip()
            if v.startswith("sk-") and len(v) > 20:
                pham.append(f"{path.relative_to(PKG)}:{lineno} (khoá OpenAI?)")
            if v.startswith("AQ.") and len(v) > 20:
                pham.append(f"{path.relative_to(PKG)}:{lineno} (khoá Google?)")
    assert not pham, "Nghi có khoá API trong mã nguồn:\n  " + "\n  ".join(pham)


# ---------- extras trong pyproject ---------------------------------------
# Đọc metadata của bản ĐÃ CÀI, không đọc chữ trong pyproject.toml: cái người
# dùng thật sự nhận được là metadata, và tomllib chỉ có từ Python 3.11 trong khi
# framework này đỡ tới 3.10.
DIST = "prompt-tuning-framework"


def _extras():
    md = importlib.metadata.metadata(DIST)
    return set(md.get_all("Provides-Extra") or [])


def _requires_for(extra: str):
    out = []
    for req in importlib.metadata.requires(DIST) or []:
        if f'extra == "{extra}"' in req or f"extra == '{extra}'" in req:
            out.append(req.split(";")[0].strip())
    return out


# [autoprompt] đứng ngoài [all] là CỐ Ý, không phải quên: langchain<0.3 của nó
# kéo langchain-google-genai từ 4.x tụt về 1.x và làm pip dò ngược hàng chục
# phiên bản. Lý do đầy đủ nằm trong pyproject.toml.
NGOAI_ALL = {"autoprompt"}


def test_all_extra_covers_every_other_extra():
    """[all] phải bao mọi extras khác, trừ danh sách loại trừ đã ghi rõ lý do.

    Bản chép tay đã từng sai: nó lấy easydict của [autoprompt] nhưng bỏ quên
    langchain, nên `pip install [all]` báo thành công trong khi plugin AutoPrompt
    không chạy được — không có lỗi, không có cảnh báo, chỉ là một test bị skip
    lặng lẽ. Nay [all] tự tham chiếu các extras kia, và test này canh cho nó luôn
    tham chiếu ĐỦ: thêm extras mới mà quên [all] là đỏ ngay.
    """
    khac = _extras() - {"all"} - NGOAI_ALL
    assert khac, "Không đọc được extras — gói đã cài chưa?"

    goi = _requires_for("all")
    assert len(goi) == 1, f"[all] nên tự tham chiếu bằng 1 dòng, đang là: {goi}"

    m = re.search(r"\[([^\]]+)\]", goi[0])
    assert m, f"[all] không tham chiếu extras nào: {goi[0]!r}"
    duoc_bao = {x.strip() for x in m.group(1).split(",")}

    thieu = khac - duoc_bao
    assert not thieu, (
        f"[all] bỏ sót extras: {sorted(thieu)}. "
        f"Sửa pyproject.toml: all = [\"{DIST}[{','.join(sorted(khac))}]\"]"
    )

    thua = duoc_bao & NGOAI_ALL
    assert not thua, (
        f"[all] kéo theo {sorted(thua)} — extras này xung khắc với phần còn lại "
        f"và sẽ hạ cấp provider. Xem lý do trong pyproject.toml."
    )


def test_autoprompt_extra_is_the_only_thing_pinning_old_langchain():
    """Cái pin langchain<0.3 chỉ được phép nằm trong [autoprompt].

    Nó đến từ upstream AutoPrompt (requirements.txt ghim langchain==0.2.7). Lọt
    sang dependencies gốc hay sang extras khác là kéo cả môi trường về đời 2024,
    và người dùng langchain>=0.3 sẽ không cài nổi framework này.
    """
    for req in importlib.metadata.requires(DIST) or []:
        if not re.match(r"^langchain\b(?!-)", req):
            continue
        if "<0.3" in req or "<0,3" in req:
            assert 'extra == "autoprompt"' in req, (
                f"Pin langchain cũ lọt ra ngoài [autoprompt]: {req!r}"
            )


def test_two_providers_are_installed_without_any_extra():
    """Cài trần (không extras) phải dùng được ngay cả Gemini lẫn OpenAI.

    Nếu ai đó đẩy hai provider ngược về extras, `pip install prompt_tuning_framework/`
    vẫn báo thành công nhưng sập lúc gọi model — sai lệch chỉ lộ ra khi tốn tiền
    gọi API thật. Đây là chốt chặn cho điều đó.
    """
    goc = [r.split(";")[0].strip() for r in (importlib.metadata.requires(DIST) or [])
           if "extra ==" not in r]
    ten = {re.split(r"[<>=!\[ ]", r, 1)[0].lower() for r in goc}
    for phai_co in ("langchain-google-genai", "langchain-openai"):
        assert phai_co in ten, (
            f"{phai_co} không nằm trong dependencies bắt buộc — cài trần sẽ "
            f"không gọi được model. Đang có: {sorted(ten)}"
        )
