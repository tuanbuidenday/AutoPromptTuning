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


class FakeChain:
    """Giả ChainWrapper của AutoPrompt: ghi lại input, trả về thứ được cài sẵn."""

    def __init__(self, tra_ve):
        self.tra_ve = tra_ve
        self.da_nhan = None

    def invoke(self, chain_input):
        self.da_nhan = chain_input
        return self.tra_ve


def _adapter(tra_ve, provider="google", max_errors=4, labels=("Yes", "No")):
    o = object.__new__(AutoPromptOptimizer)
    o.chain = FakeChain(tra_ve)
    o.labels = list(labels)
    o.max_errors = max_errors
    o._provider = provider
    o.model = "model-gia"
    return o


def _result(score=50.0, n_loi=2):
    results = [
        SampleResult(sample=Sample(id=i, text=f"ticket {i}", label="Yes"),
                     predicted="No", expected="Yes", correct=False)
        for i in range(n_loi)
    ]
    return EvalResult(score=score, results=results)


# ---------- đăng ký plugin ----------------------------------------------
def test_registered_in_registry():
    """Luận điểm 'AutoPrompt chỉ là một plugin' nằm ở đúng dòng này."""
    assert "autoprompt" in available("optimizer")
    assert get("optimizer", "autoprompt") is AutoPromptOptimizer


def test_peer_of_builtin_optimizer():
    """Nó phải đứng ngang hàng llm_rewrite, không phải trường hợp đặc biệt."""
    assert {"autoprompt", "llm_rewrite"} <= set(available("optimizer"))


# ---------- quirk của Gemini: tool-call trả về list ----------------------
def test_unwraps_gemini_list_tool_call():
    """Gemini trả [{'args': {...}}] chứ không trả thẳng dict."""
    o = _adapter([{"args": {"prompt": "prompt moi"}}])
    assert o.propose("cu", _result()) == "prompt moi"


def test_other_provider_does_not_unwrap_list():
    """Chỉ google mới có quirk này; bóc nhầm ở provider khác là sai."""
    o = _adapter({"prompt": "prompt moi"}, provider="openai")
    assert o.propose("cu", _result()) == "prompt moi"


def test_google_still_accepts_plain_dict():
    o = _adapter({"prompt": "prompt moi"})
    assert o.propose("cu", _result()) == "prompt moi"


def test_multi_element_list_not_unwrapped():
    o = _adapter([{"args": {"prompt": "a"}}, {"args": {"prompt": "b"}}])
    assert o.propose("cu", _result()) == ""


# ---------- ChainWrapper nuốt lỗi và trả None ---------------------------
def test_chain_returning_none_gives_empty_string():
    """utils/llm_chain.py:67 nuốt MỌI exception rồi trả None.

    Adapter phải trả "" để tuner dừng gọn, thay vì nổ TypeError như chính
    AutoPrompt gốc vẫn nổ ('NoneType' object is not subscriptable).
    """
    o = _adapter(None)
    assert o.propose("cu", _result()) == ""


def test_missing_prompt_key_gives_empty_string():
    o = _adapter({"khong_phai_prompt": "x"})
    assert o.propose("cu", _result()) == ""


def test_empty_list_gives_empty_string():
    o = _adapter([])
    assert o.propose("cu", _result()) == ""


def test_strips_surrounding_whitespace():
    o = _adapter({"prompt": "  co khoang trang  \n"})
    assert o.propose("cu", _result()) == "co khoang trang"


# ---------- nội dung gửi cho meta-chain ---------------------------------
def test_sends_all_4_fields_autoprompt_requires():
    o = _adapter({"prompt": "x"})
    o.propose("cu", _result(), task_description="Phan loai ticket")
    nhan = o.chain.da_nhan
    assert set(nhan) == {"task_description", "history", "error_analysis", "labels"}
    assert nhan["task_description"] == "Phan loai ticket"
    assert json.loads(nhan["labels"]) == ["Yes", "No"]


def test_runs_without_task_description():
    o = _adapter({"prompt": "x"})
    o.propose("cu", _result())
    assert o.chain.da_nhan["task_description"] == "(not provided)"


def test_error_analysis_includes_failed_samples():
    o = _adapter({"prompt": "x"})
    o.propose("cu", _result(score=42.0, n_loi=2))
    ea = o.chain.da_nhan["error_analysis"]
    assert "42.0/100" in ea
    assert "ticket 0" in ea and "ticket 1" in ea
    assert "Predicted: No" in ea and "Correct: Yes" in ea


def test_error_analysis_respects_max_errors():
    """Nhồi hết lỗi vào meta-prompt sẽ phình token và tốn tiền vô ích."""
    o = _adapter({"prompt": "x"}, max_errors=2)
    o.propose("cu", _result(n_loi=10))
    ea = o.chain.da_nhan["error_analysis"]
    assert "ticket 1" in ea
    assert "ticket 2" not in ea


def test_proposes_even_with_no_failures():
    """Prompt đã đúng hết vẫn phải có đường đi tiếp, không được gửi rỗng."""
    o = _adapter({"prompt": "x"})
    o.propose("cu", EvalResult(score=100.0, results=[]))
    assert "no mistakes" in o.chain.da_nhan["error_analysis"]


# ---------- lịch sử ------------------------------------------------------
def test_history_uses_autoprompt_format():
    o = _adapter({"prompt": "x"})
    lich_su = [PromptVersion(version=0, text="p0", score=50.0),
               PromptVersion(version=1, text="p1", score=70.0)]
    o.propose("cu", _result(), history=lich_su)
    h = o.chain.da_nhan["history"]
    assert "##Prompt Score: 50.00" in h and "##Prompt Score: 70.00" in h
    assert "p0" in h and "p1" in h


def test_history_skips_unscored_versions():
    o = _adapter({"prompt": "x"})
    lich_su = [PromptVersion(version=0, text="da cham", score=50.0),
               PromptVersion(version=1, text="chua cham", score=None)]
    o.propose("cu", _result(), history=lich_su)
    assert "chua cham" not in o.chain.da_nhan["history"]


def test_history_keeps_only_4_most_recent():
    o = _adapter({"prompt": "x"})
    lich_su = [PromptVersion(version=i, text=f"p{i}", score=float(i))
               for i in range(6)]
    o.propose("cu", _result(), history=lich_su)
    h = o.chain.da_nhan["history"]
    assert "p0" not in h and "p1" not in h
    assert "p5" in h


def test_no_history_writes_none():
    o = _adapter({"prompt": "x"})
    o.propose("cu", _result())
    assert o.chain.da_nhan["history"] == "(none)"


# ---------- tích hợp: dựng thật, cần deps của AutoPrompt ----------------
def test_works_for_real_when_deps_installed():
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


# ---------- thiếu repo AutoPrompt thì phải nói rõ ------------------------
def test_missing_autoprompt_repo_says_how_to_fix(monkeypatch, tmp_path):
    """Người cài bằng pip nhận được hướng dẫn, không phải "No module named 'utils'".

    Plugin cần cả repo AutoPrompt trên đĩa, mà pip không cài được: AutoPrompt
    không có trên PyPI. Người cài `pip install prompt-tuning-framework` sẽ có
    _REPO_ROOT trỏ vào site-packages, nơi không hề có utils/llm_chain.py. Lỗi
    trần không cho họ manh mối nào để tự thoát.
    """
    from prompt_tuning_framework.components.optimizers import autoprompt_optimizer as m
    monkeypatch.setattr(m, "_REPO_ROOT", tmp_path)   # thư mục rỗng = không có repo

    with pytest.raises(ModuleNotFoundError) as e:
        m.AutoPromptOptimizer(labels=["Yes", "No"])

    loi = str(e.value)
    assert "git clone" in loi, "Phải chỉ ra cách lấy repo về"
    assert "llm_rewrite" in loi, "Phải nêu lối thoát cho người không cần AutoPrompt"
    assert str(tmp_path) in loi, "Phải nói rõ nó đang tìm ở đâu"


def test_missing_repo_check_runs_before_importing_utils(monkeypatch, tmp_path):
    """Kiểm tra repo phải chạy TRƯỚC import, nếu không thông báo tử tế vô dụng.

    Nếu để `from utils.llm_chain import ...` chạy trước, Python ném
    ModuleNotFoundError thô ngay tại đó và hướng dẫn không bao giờ tới tay ai.
    """
    from prompt_tuning_framework.components.optimizers import autoprompt_optimizer as m
    monkeypatch.setattr(m, "_REPO_ROOT", tmp_path)

    with pytest.raises(ModuleNotFoundError) as e:
        m.AutoPromptOptimizer(labels=["Yes", "No"])
    assert "No module named 'utils'" not in str(e.value), (
        "Lỗi trần lọt ra ngoài — phần kiểm tra repo đang chạy sau import"
    )
