"""Test lớp LLM: model mặc định theo provider + thứ tự tìm API key.

Chạy hoàn toàn offline — không gọi API thật (build_llm bị monkeypatch).
"""
import pytest

from prompt_tuning_framework import llm
from prompt_tuning_framework.components.executors import llm_executor
from prompt_tuning_framework.components.optimizers import llm_rewrite_optimizer
from prompt_tuning_framework.llm import DEFAULT_MODELS, default_model, resolve_api_key


@pytest.fixture
def bat_build_llm(monkeypatch):
    """Chặn build_llm, ghi lại tham số thay vì gọi API thật."""
    captured = {}

    def fake_build_llm(provider=None, model=None, temperature=None, api_key=None, **kw):
        captured.update(provider=provider, model=model)
        return object()

    for mod in (llm_executor, llm_rewrite_optimizer):
        monkeypatch.setattr(mod, "build_llm", fake_build_llm)
    return captured


# ---------------- Model mặc định theo provider ----------------

def test_default_model_cua_google():
    assert default_model("google", "executor") == "gemini-3.1-flash-lite"
    assert default_model("google", "optimizer") == "gemini-3.5-flash"


def test_default_model_cua_openai_la_ban_re():
    assert default_model("openai", "executor") == "gpt-4o-mini"
    assert default_model("openai", "optimizer") == "gpt-4o-mini"


def test_default_model_khong_phan_biet_hoa_thuong():
    assert default_model("OpenAI", "executor") == default_model("openai", "executor")


@pytest.mark.parametrize("provider,cam", [("openai", "gemini"), ("google", "gpt")])
def test_model_mac_dinh_khong_lan_sang_provider_khac(provider, cam):
    """Chốt chặn bug: đổi provider mà model mặc định vẫn của hãng kia."""
    for role in ("executor", "optimizer"):
        assert cam not in default_model(provider, role).lower()


def test_moi_provider_deu_co_du_2_vai():
    for provider, roles in DEFAULT_MODELS.items():
        assert set(roles) == {"executor", "optimizer"}, provider


def test_provider_la_bao_loi_ro_rang():
    with pytest.raises(ValueError, match="Provider hỗ trợ"):
        default_model("anthropic", "executor")


# ---------------- Component phải bám theo provider ----------------

def test_executor_openai_khong_gui_model_gemini(bat_build_llm):
    """Trước đây LLMExecutor(provider='openai') vẫn gửi gemini-2.5-flash-lite."""
    ex = llm_executor.LLMExecutor(provider="openai", labels=["Yes", "No"])
    assert bat_build_llm["model"] == "gpt-4o-mini"
    assert ex.model == "gpt-4o-mini"


def test_optimizer_openai_khong_gui_model_gemini(bat_build_llm):
    opt = llm_rewrite_optimizer.LLMRewriteOptimizer(provider="openai", labels=["Yes", "No"])
    assert bat_build_llm["model"] == "gpt-4o-mini"
    assert opt.model == "gpt-4o-mini"


def test_executor_mac_dinh_van_la_google(bat_build_llm):
    llm_executor.LLMExecutor(labels=["Yes", "No"])
    assert bat_build_llm["provider"] == "google"
    assert bat_build_llm["model"] == "gemini-3.1-flash-lite"


def test_model_truyen_tay_duoc_uu_tien(bat_build_llm):
    llm_executor.LLMExecutor(provider="openai", model="gpt-4o", labels=["Yes"])
    assert bat_build_llm["model"] == "gpt-4o"


# ---------------- Thứ tự tìm API key ----------------
# Ưu tiên: tham số -> biến môi trường -> llm_env.local.yml -> llm_env.yml

@pytest.fixture
def env_sach(monkeypatch, tmp_path):
    """Cô lập khỏi biến môi trường và file llm_env* THẬT trên máy.

    Không có fixture này, test sẽ đọc nhầm khoá thật của lập trình viên và
    xanh/đỏ tuỳ máy.
    """
    for var in ("OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm, "_LLM_ENV", tmp_path / "khong-co.yml")
    monkeypatch.setattr(llm, "_LLM_ENV_LOCAL", tmp_path / "khong-co-local.yml")
    return tmp_path


def _viet_env(path, key):
    path.write_text(f"openai:\n    OPENAI_API_KEY: '{key}'\n", encoding="utf-8")
    return path


def test_api_key_uu_tien_tham_so_cao_nhat(env_sach, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-tu-env")
    assert resolve_api_key("openai", "sk-tu-tham-so") == "sk-tu-tham-so"


def test_api_key_doc_tu_bien_moi_truong(env_sach, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-tu-env")
    assert resolve_api_key("openai") == "sk-tu-env"


def test_api_key_doc_tu_llm_env_yml(env_sach, monkeypatch):
    monkeypatch.setattr(llm, "_LLM_ENV", _viet_env(env_sach / "a.yml", "sk-tu-file"))
    assert llm.resolve_api_key("openai") == "sk-tu-file"


def test_api_key_doc_tu_file_local(env_sach, monkeypatch):
    """Khoá thật sống ở llm_env.local.yml (đã gitignore)."""
    monkeypatch.setattr(llm, "_LLM_ENV_LOCAL",
                        _viet_env(env_sach / "b.yml", "sk-tu-local"))
    assert llm.resolve_api_key("openai") == "sk-tu-local"


def test_file_local_uu_tien_hon_llm_env_yml(env_sach, monkeypatch):
    """llm_env.yml trong Git chỉ là template rỗng -> .local phải thắng."""
    monkeypatch.setattr(llm, "_LLM_ENV", _viet_env(env_sach / "a.yml", "sk-template"))
    monkeypatch.setattr(llm, "_LLM_ENV_LOCAL", _viet_env(env_sach / "b.yml", "sk-that"))
    assert llm.resolve_api_key("openai") == "sk-that"


def test_bien_moi_truong_uu_tien_hon_moi_file(env_sach, monkeypatch):
    monkeypatch.setattr(llm, "_LLM_ENV", _viet_env(env_sach / "a.yml", "sk-template"))
    monkeypatch.setattr(llm, "_LLM_ENV_LOCAL", _viet_env(env_sach / "b.yml", "sk-that"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-tu-env")
    assert llm.resolve_api_key("openai") == "sk-tu-env"


def test_khong_co_key_o_dau_thi_tra_none(env_sach):
    assert llm.resolve_api_key("openai") is None


def test_key_rong_trong_file_coi_nhu_khong_co(env_sach, monkeypatch):
    """llm_env.yml trong Git có sẵn slot rỗng -> không được trả về chuỗi rỗng."""
    monkeypatch.setattr(llm, "_LLM_ENV", _viet_env(env_sach / "a.yml", ""))
    assert llm.resolve_api_key("openai") is None


def test_local_rong_thi_lui_ve_llm_env_yml(env_sach, monkeypatch):
    monkeypatch.setattr(llm, "_LLM_ENV", _viet_env(env_sach / "a.yml", "sk-template"))
    monkeypatch.setattr(llm, "_LLM_ENV_LOCAL", _viet_env(env_sach / "b.yml", ""))
    assert llm.resolve_api_key("openai") == "sk-template"


def test_file_yaml_hong_khong_lam_sap(env_sach, monkeypatch):
    """File YAML lỗi cú pháp -> trả None, không ném exception ra ngoài."""
    xau = env_sach / "hong.yml"
    xau.write_text("openai: [khong dong ngoac\n", encoding="utf-8")
    monkeypatch.setattr(llm, "_LLM_ENV_LOCAL", xau)
    assert llm.resolve_api_key("openai") is None


def test_build_llm_bao_loi_voi_provider_la():
    with pytest.raises(ValueError):
        llm.build_llm(provider="cohere", model="x")
