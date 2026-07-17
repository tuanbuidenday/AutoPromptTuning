"""Test lớp LLM: model mặc định theo provider + thứ tự tìm API key.

Chạy hoàn toàn offline — không gọi API thật (build_llm bị monkeypatch).
"""
import pytest

from prompt_tuning_framework import llm
from prompt_tuning_framework.components.executors import llm_executor
from prompt_tuning_framework.components.optimizers import llm_rewrite_optimizer
from prompt_tuning_framework.llm import DEFAULT_MODELS, default_model, resolve_api_key


@pytest.fixture
def patch_build_llm(monkeypatch):
    """Chặn build_llm, ghi lại tham số thay vì gọi API thật."""
    captured = {}

    def fake_build_llm(provider=None, model=None, temperature=None, api_key=None, **kw):
        captured.update(provider=provider, model=model)
        return object()

    for mod in (llm_executor, llm_rewrite_optimizer):
        monkeypatch.setattr(mod, "build_llm", fake_build_llm)
    return captured


# ---------------- Model mặc định theo provider ----------------

def test_google_default_model():
    assert default_model("google", "executor") == "gemini-3.1-flash-lite"
    assert default_model("google", "optimizer") == "gemini-3.5-flash"


def test_openai_default_model_is_the_cheap_one():
    assert default_model("openai", "executor") == "gpt-4o-mini"
    assert default_model("openai", "optimizer") == "gpt-4o-mini"


def test_default_model_lookup_is_case_insensitive():
    assert default_model("OpenAI", "executor") == default_model("openai", "executor")


@pytest.mark.parametrize("provider,cam", [("openai", "gemini"), ("google", "gpt")])
def test_default_model_does_not_bleed_across_providers(provider, cam):
    """Chốt chặn bug: đổi provider mà model mặc định vẫn của hãng kia."""
    for role in ("executor", "optimizer"):
        assert cam not in default_model(provider, role).lower()


def test_every_provider_has_both_roles():
    for provider, roles in DEFAULT_MODELS.items():
        assert set(roles) == {"executor", "optimizer"}, provider


def test_unknown_provider_errors_clearly():
    with pytest.raises(ValueError, match="Provider hỗ trợ"):
        default_model("anthropic", "executor")


# ---------------- Component phải bám theo provider ----------------

def test_openai_executor_does_not_send_gemini_model(patch_build_llm):
    """Trước đây LLMExecutor(provider='openai') vẫn gửi gemini-2.5-flash-lite."""
    ex = llm_executor.LLMExecutor(provider="openai", labels=["Yes", "No"])
    assert patch_build_llm["model"] == "gpt-4o-mini"
    assert ex.model == "gpt-4o-mini"


def test_openai_optimizer_does_not_send_gemini_model(patch_build_llm):
    opt = llm_rewrite_optimizer.LLMRewriteOptimizer(provider="openai", labels=["Yes", "No"])
    assert patch_build_llm["model"] == "gpt-4o-mini"
    assert opt.model == "gpt-4o-mini"


def test_executor_still_defaults_to_google(patch_build_llm):
    llm_executor.LLMExecutor(labels=["Yes", "No"])
    assert patch_build_llm["provider"] == "google"
    assert patch_build_llm["model"] == "gemini-3.1-flash-lite"


def test_explicit_model_takes_precedence(patch_build_llm):
    llm_executor.LLMExecutor(provider="openai", model="gpt-4o", labels=["Yes"])
    assert patch_build_llm["model"] == "gpt-4o"


# ---------------- Thứ tự tìm API key ----------------
# Ưu tiên: tham số -> biến môi trường -> llm_env.local.yml -> llm_env.yml

@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """Cô lập khỏi biến môi trường và file llm_env* THẬT trên máy.

    Không có fixture này, test sẽ đọc nhầm khoá thật của lập trình viên và
    xanh/đỏ tuỳ máy.
    """
    for var in ("OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm, "_LLM_ENV", tmp_path / "khong-co.yml")
    monkeypatch.setattr(llm, "_LLM_ENV_LOCAL", tmp_path / "khong-co-local.yml")
    return tmp_path


def _write_env(path, key):
    path.write_text(f"openai:\n    OPENAI_API_KEY: '{key}'\n", encoding="utf-8")
    return path


def test_api_key_argument_has_highest_precedence(clean_env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-tu-env")
    assert resolve_api_key("openai", "sk-tu-tham-so") == "sk-tu-tham-so"


def test_api_key_read_from_env_var(clean_env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-tu-env")
    assert resolve_api_key("openai") == "sk-tu-env"


def test_api_key_read_from_llm_env_yml(clean_env, monkeypatch):
    monkeypatch.setattr(llm, "_LLM_ENV", _write_env(clean_env / "a.yml", "sk-tu-file"))
    assert llm.resolve_api_key("openai") == "sk-tu-file"


def test_api_key_read_from_local_file(clean_env, monkeypatch):
    """Khoá thật sống ở llm_env.local.yml (đã gitignore)."""
    monkeypatch.setattr(llm, "_LLM_ENV_LOCAL",
                        _write_env(clean_env / "b.yml", "sk-tu-local"))
    assert llm.resolve_api_key("openai") == "sk-tu-local"


def test_local_file_beats_llm_env_yml(clean_env, monkeypatch):
    """llm_env.yml trong Git chỉ là template rỗng -> .local phải thắng."""
    monkeypatch.setattr(llm, "_LLM_ENV", _write_env(clean_env / "a.yml", "sk-template"))
    monkeypatch.setattr(llm, "_LLM_ENV_LOCAL", _write_env(clean_env / "b.yml", "sk-that"))
    assert llm.resolve_api_key("openai") == "sk-that"


def test_env_var_beats_every_file(clean_env, monkeypatch):
    monkeypatch.setattr(llm, "_LLM_ENV", _write_env(clean_env / "a.yml", "sk-template"))
    monkeypatch.setattr(llm, "_LLM_ENV_LOCAL", _write_env(clean_env / "b.yml", "sk-that"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-tu-env")
    assert llm.resolve_api_key("openai") == "sk-tu-env"


def test_no_key_anywhere_returns_none(clean_env):
    assert llm.resolve_api_key("openai") is None


def test_empty_key_in_file_treated_as_absent(clean_env, monkeypatch):
    """llm_env.yml trong Git có sẵn slot rỗng -> không được trả về chuỗi rỗng."""
    monkeypatch.setattr(llm, "_LLM_ENV", _write_env(clean_env / "a.yml", ""))
    assert llm.resolve_api_key("openai") is None


def test_empty_local_falls_back_to_llm_env_yml(clean_env, monkeypatch):
    monkeypatch.setattr(llm, "_LLM_ENV", _write_env(clean_env / "a.yml", "sk-template"))
    monkeypatch.setattr(llm, "_LLM_ENV_LOCAL", _write_env(clean_env / "b.yml", ""))
    assert llm.resolve_api_key("openai") == "sk-template"


def test_corrupt_yaml_does_not_crash(clean_env, monkeypatch):
    """File YAML lỗi cú pháp -> trả None, không ném exception ra ngoài."""
    xau = clean_env / "hong.yml"
    xau.write_text("openai: [khong dong ngoac\n", encoding="utf-8")
    monkeypatch.setattr(llm, "_LLM_ENV_LOCAL", xau)
    assert llm.resolve_api_key("openai") is None


def test_build_llm_errors_on_unknown_provider():
    with pytest.raises(ValueError):
        llm.build_llm(provider="cohere", model="x")
