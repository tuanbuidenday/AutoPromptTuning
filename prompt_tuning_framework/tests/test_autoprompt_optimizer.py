"""Kiểm thử AutoPromptOptimizer — adapter cắm engine AutoPrompt vào framework.

Plugin này là BẰNG CHỨNG cho luận điểm trung tâm của cả dự án: "AutoPrompt chỉ là
MỘT optimizer cắm vào, không phải lõi". Nó từng không có test nào.

Vì sao không gọi __init__ thật ở đây: __init__ dựng ChainWrapper của AutoPrompt,
kéo theo easydict + langchain + langchain_google_genai. CI chỉ cài extras [test]
nên các gói đó không có. Dựng đối tượng bằng object.__new__ rồi tiêm chain giả
cho phép kiểm thử ĐÚNG phần logic của adapter (bóc tool-call, xử lý lỗi, định
dạng meta-prompt) ở mọi môi trường. Phần gọi thật nằm ở test tích hợp cuối file,
tự bỏ qua khi thiếu deps.
"""
import json

import pytest

from prompt_tuning_framework import EvalResult, PromptVersion, Sample, SampleResult
from prompt_tuning_framework.components.optimizers.autoprompt_optimizer import \
    AutoPromptOptimizer
from prompt_tuning_framework.core.registry import available, get


class ChainGia:
    """Giả ChainWrapper của AutoPrompt: ghi lại input, trả về thứ được cài sẵn."""

    def __init__(self, tra_ve):
        self.tra_ve = tra_ve
        self.da_nhan = None

    def invoke(self, chain_input):
        self.da_nhan = chain_input
        return self.tra_ve


def _adapter(tra_ve, provider="google", max_errors=4, labels=("Yes", "No")):
    o = object.__new__(AutoPromptOptimizer)
    o.chain = ChainGia(tra_ve)
    o.labels = list(labels)
    o.max_errors = max_errors
    o._provider = provider
    o.model = "model-gia"
    return o


def _kq(score=50.0, n_loi=2):
    results = [
        SampleResult(sample=Sample(id=i, text=f"ticket {i}", label="Yes"),
                     predicted="No", expected="Yes", correct=False)
        for i in range(n_loi)
    ]
    return EvalResult(score=score, results=results)


# ---------- đăng ký plugin ----------------------------------------------
def test_da_dang_ky_vao_registry():
    """Luận điểm 'AutoPrompt chỉ là một plugin' nằm ở đúng dòng này."""
    assert "autoprompt" in available("optimizer")
    assert get("optimizer", "autoprompt") is AutoPromptOptimizer


def test_ngang_hang_voi_optimizer_nha():
    """Nó phải đứng ngang hàng llm_rewrite, không phải trường hợp đặc biệt."""
    assert {"autoprompt", "llm_rewrite"} <= set(available("optimizer"))


# ---------- quirk của Gemini: tool-call trả về list ----------------------
def test_boc_tool_call_dang_list_cua_gemini():
    """Gemini trả [{'args': {...}}] chứ không trả thẳng dict."""
    o = _adapter([{"args": {"prompt": "prompt moi"}}])
    assert o.propose("cu", _kq()) == "prompt moi"


def test_provider_khac_khong_boc_list():
    """Chỉ google mới có quirk này; bóc nhầm ở provider khác là sai."""
    o = _adapter({"prompt": "prompt moi"}, provider="openai")
    assert o.propose("cu", _kq()) == "prompt moi"


def test_google_van_nhan_dict_thuong():
    o = _adapter({"prompt": "prompt moi"})
    assert o.propose("cu", _kq()) == "prompt moi"


def test_list_nhieu_phan_tu_thi_khong_boc():
    o = _adapter([{"args": {"prompt": "a"}}, {"args": {"prompt": "b"}}])
    assert o.propose("cu", _kq()) == ""


# ---------- ChainWrapper nuốt lỗi và trả None ---------------------------
def test_chain_tra_none_thi_tra_chuoi_rong():
    """utils/llm_chain.py:67 nuốt MỌI exception rồi trả None.

    Adapter phải trả "" để tuner dừng gọn, thay vì nổ TypeError như chính
    AutoPrompt gốc vẫn nổ ('NoneType' object is not subscriptable).
    """
    o = _adapter(None)
    assert o.propose("cu", _kq()) == ""


def test_thieu_khoa_prompt_thi_tra_chuoi_rong():
    o = _adapter({"khong_phai_prompt": "x"})
    assert o.propose("cu", _kq()) == ""


def test_list_rong_thi_tra_chuoi_rong():
    o = _adapter([])
    assert o.propose("cu", _kq()) == ""


def test_cat_khoang_trang_thua():
    o = _adapter({"prompt": "  co khoang trang  \n"})
    assert o.propose("cu", _kq()) == "co khoang trang"


# ---------- nội dung gửi cho meta-chain ---------------------------------
def test_gui_du_4_truong_autoprompt_yeu_cau():
    o = _adapter({"prompt": "x"})
    o.propose("cu", _kq(), task_description="Phan loai ticket")
    nhan = o.chain.da_nhan
    assert set(nhan) == {"task_description", "history", "error_analysis", "labels"}
    assert nhan["task_description"] == "Phan loai ticket"
    assert json.loads(nhan["labels"]) == ["Yes", "No"]


def test_khong_co_task_description_thi_van_chay():
    o = _adapter({"prompt": "x"})
    o.propose("cu", _kq())
    assert o.chain.da_nhan["task_description"] == "(not provided)"


def test_error_analysis_chua_ca_sai():
    o = _adapter({"prompt": "x"})
    o.propose("cu", _kq(score=42.0, n_loi=2))
    ea = o.chain.da_nhan["error_analysis"]
    assert "42.0/100" in ea
    assert "ticket 0" in ea and "ticket 1" in ea
    assert "Predicted: No" in ea and "Correct: Yes" in ea


def test_error_analysis_ton_trong_max_errors():
    """Nhồi hết lỗi vào meta-prompt sẽ phình token và tốn tiền vô ích."""
    o = _adapter({"prompt": "x"}, max_errors=2)
    o.propose("cu", _kq(n_loi=10))
    ea = o.chain.da_nhan["error_analysis"]
    assert "ticket 1" in ea
    assert "ticket 2" not in ea


def test_khong_co_ca_sai_thi_van_de_xuat_duoc():
    """Prompt đã đúng hết vẫn phải có đường đi tiếp, không được gửi rỗng."""
    o = _adapter({"prompt": "x"})
    o.propose("cu", EvalResult(score=100.0, results=[]))
    assert "no mistakes" in o.chain.da_nhan["error_analysis"]


# ---------- lịch sử ------------------------------------------------------
def test_history_dung_dinh_dang_cua_autoprompt():
    o = _adapter({"prompt": "x"})
    lich_su = [PromptVersion(version=0, text="p0", score=50.0),
               PromptVersion(version=1, text="p1", score=70.0)]
    o.propose("cu", _kq(), history=lich_su)
    h = o.chain.da_nhan["history"]
    assert "##Prompt Score: 50.00" in h and "##Prompt Score: 70.00" in h
    assert "p0" in h and "p1" in h


def test_history_bo_qua_ban_chua_cham_diem():
    o = _adapter({"prompt": "x"})
    lich_su = [PromptVersion(version=0, text="da cham", score=50.0),
               PromptVersion(version=1, text="chua cham", score=None)]
    o.propose("cu", _kq(), history=lich_su)
    assert "chua cham" not in o.chain.da_nhan["history"]


def test_history_chi_lay_4_ban_gan_nhat():
    o = _adapter({"prompt": "x"})
    lich_su = [PromptVersion(version=i, text=f"p{i}", score=float(i))
               for i in range(6)]
    o.propose("cu", _kq(), history=lich_su)
    h = o.chain.da_nhan["history"]
    assert "p0" not in h and "p1" not in h
    assert "p5" in h


def test_khong_co_history_thi_ghi_none():
    o = _adapter({"prompt": "x"})
    o.propose("cu", _kq())
    assert o.chain.da_nhan["history"] == "(none)"


# ---------- tích hợp: dựng thật, cần deps của AutoPrompt ----------------
def test_dung_that_duoc_khi_co_deps():
    """Kiểm chứng __init__ thật: EasyDict, đọc file meta-prompt, dựng ChainWrapper.

    Bỏ qua nếu thiếu easydict/langchain (CI chỉ cài extras [test]) hoặc thiếu API
    key. Không gọi LLM — chỉ dựng đối tượng.
    """
    pytest.importorskip("easydict")
    pytest.importorskip("langchain")
    from prompt_tuning_framework.llm import resolve_api_key
    if not resolve_api_key("google"):
        pytest.skip("chưa có GOOGLE_API_KEY")

    o = AutoPromptOptimizer(labels=["Yes", "No"])
    assert o.chain is not None
    assert o._provider == "google"
    # Tên model không được hardcode trong adapter — phải lấy từ llm.DEFAULT_MODELS.
    from prompt_tuning_framework.llm import default_model
    assert o.model == default_model("google", "optimizer")
