"""Lớp mỏng trừu tượng hoá LLM provider cho framework.

Tách riêng để đổi provider (Google / OpenAI) mà không đụng vào component.
"""
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

_DIST = "prompt-tuning-framework"
_REPO_ROOT = Path(__file__).resolve().parent.parent
# llm_env.yml nằm trong Git nhưng chỉ là template toàn giá trị rỗng.
# Khoá THẬT để ở llm_env.local.yml — đã .gitignore nên không thể lọt lên repo.
_LLM_ENV = _REPO_ROOT / "config" / "llm_env.yml"
_LLM_ENV_LOCAL = _REPO_ROOT / "config" / "llm_env.local.yml"

# Model mặc định theo provider — mặc định phải RẺ, người dùng không nên bị đốt
# tiền chỉ vì quên truyền --model. Tách 2 vai vì chi phí rất lệch nhau:
#   executor  chạy prompt trên MỌI sample, MỖI vòng   -> nơi tốn tiền nhất
#   optimizer chỉ chạy 1 lần mỗi vòng                 -> nâng model ở đây rất rẻ
# Google đã ngừng cấp họ 2.5 cho tài khoản mới (gọi vào là 404), dù list_models
# vẫn liệt kê chúng -> luôn thử gọi thật, đừng tin danh sách.
# Không dùng tên có '-latest': nó âm thầm trỏ sang model đắt hơn.
DEFAULT_MODELS = {
    "google": {"executor": "gemini-3.1-flash-lite", "optimizer": "gemini-3.5-flash"},
    "openai": {"executor": "gpt-4o-mini", "optimizer": "gpt-4o-mini"},
}


def default_model(provider: str, role: str = "executor") -> str:
    """Model mặc định của provider theo vai ('executor' | 'optimizer').

    Có hàm này để không xảy ra chuyện gửi tên model Gemini sang OpenAI khi người
    dùng đổi provider mà quên đổi model.
    """
    try:
        return DEFAULT_MODELS[provider.lower()][role]
    except KeyError:
        raise ValueError(
            f"Không có model mặc định cho provider={provider!r}, role={role!r}. "
            f"Provider hỗ trợ: {', '.join(DEFAULT_MODELS)}. "
            f"Hãy chỉ định model cụ thể."
        ) from None


def _key_from_yaml(path: Path, provider: str, env_var: str) -> Optional[str]:
    """Đọc key từ một file llm_env; trả None nếu thiếu/rỗng/hỏng."""
    if not path.is_file():
        return None
    try:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return (data.get(provider) or {}).get(env_var) or None
    except Exception:
        return None


def resolve_api_key(provider: str, api_key: Optional[str] = None) -> Optional[str]:
    """Tìm API key theo thứ tự ưu tiên:

    tham số -> biến môi trường -> config/llm_env.local.yml -> config/llm_env.yml

    Cố ý KHÔNG nhận key qua dòng lệnh: key trong argv sẽ lộ qua `ps` và bị lưu
    vào history của shell.
    """
    if api_key:
        return api_key

    provider = provider.lower()
    env_var = {"google": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY"}.get(provider)
    if not env_var:
        return None
    if os.environ.get(env_var):
        return os.environ[env_var]

    # .local trước: đó là nơi để khoá thật. llm_env.yml chỉ là template rỗng.
    for path in (_LLM_ENV_LOCAL, _LLM_ENV):
        key = _key_from_yaml(path, provider, env_var)
        if key:
            return key
    return None


@contextmanager
def _goi_y_cai_dat(extras: str, goi: str):
    """Đổi ModuleNotFoundError thành thông báo có kèm lệnh sửa.

    Lỗi trần chỉ nói "No module named 'langchain_google_genai'", không nói phải
    làm gì. Người mới không có cách nào đoán ra tên extras để cài.
    """
    try:
        yield
    except ModuleNotFoundError as e:
        if e.name and not e.name.startswith(goi.replace("-", "_")):
            raise
        raise ModuleNotFoundError(
            f"Thiếu {goi} nên không dùng được provider {extras!r}.\n"
            f"Cài bằng:  pip install '{_DIST}[{extras}]'\n"
            f"(bản cài trần vốn đã có sẵn — nếu vẫn thiếu thì môi trường đang "
            f"lỡ gỡ mất gói này)"
        ) from e


def build_llm(provider: str = "google", model: Optional[str] = None,
              temperature: float = 0.0, api_key: Optional[str] = None,
              role: str = "executor"):
    """Trả về một chat model (LangChain) theo provider.

    :param model: để None thì lấy model mặc định (rẻ) của provider.
    """
    provider = provider.lower()
    model = model or default_model(provider, role)
    key = resolve_api_key(provider, api_key)

    if provider == "google":
        with _goi_y_cai_dat("google", "langchain-google-genai"):
            from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature,
                                      google_api_key=key)
    if provider == "openai":
        with _goi_y_cai_dat("openai", "langchain-openai"):
            from langchain_openai import ChatOpenAI
        return ChatOpenAI(model_name=model, temperature=temperature, openai_api_key=key)

    raise ValueError(f"Provider chưa hỗ trợ: {provider!r} (hiện có: google, openai)")
