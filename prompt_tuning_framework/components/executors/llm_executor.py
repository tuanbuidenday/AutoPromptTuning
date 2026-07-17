"""② Executor chạy prompt bằng LLM (Google Gemini / OpenAI)."""
import concurrent.futures
import time
from typing import List, Optional

from ...core.interfaces import BaseExecutor
from ...core.registry import register
from ...core.types import Prediction, Sample
from ...llm import build_llm, default_model

DEFAULT_TEMPLATE = (
    "{prompt}\n\n"
    "Input:\n{text}\n\n"
    "Answer with the label only, no explanation."
)


@register("executor", "llm")
class LLMExecutor(BaseExecutor):
    """Chạy prompt trên từng sample và lấy nhãn LLM trả về.

    :param model: tên model; để None thì lấy model rẻ mặc định của provider
    :param provider: 'google' hoặc 'openai'
    :param labels: nếu có, ép câu trả lời về đúng một nhãn trong danh sách
    :param num_workers: số luồng song song (để 1 nếu bị rate limit)
    :param delay: nghỉ bao nhiêu giây giữa 2 lần gọi (chỉ khi num_workers=1).
        Dùng cho gói free bị giới hạn request/phút — vd free tier Gemini cho 15
        request/phút thì đặt delay=4.0. Bị rate limit sẽ làm ca lỗi, mà ca lỗi
        thì bị loại khỏi mẫu số của điểm -> điểm bị thổi phồng.
    """

    def __init__(self, model: Optional[str] = None, provider: str = "google",
                 labels: Optional[List[str]] = None, temperature: float = 0.0,
                 num_workers: int = 1, api_key: Optional[str] = None,
                 template: str = DEFAULT_TEMPLATE, delay: float = 0.0):
        self.model = model or default_model(provider, "executor")
        self.llm = build_llm(provider=provider, model=self.model,
                             temperature=temperature, api_key=api_key)
        self.labels = labels
        self.num_workers = num_workers
        self.template = template
        self.delay = delay

    def _normalize(self, raw: str) -> str:
        """Ép output về đúng nhãn trong label_schema nếu nhận ra được."""
        out = (raw or "").strip()
        if not self.labels:
            return out
        low = out.lower()
        for lb in self.labels:
            if low == lb.lower():
                return lb
        for lb in self.labels:  # LLM hay trả 'Yes.' / 'Answer: Yes'
            if lb.lower() in low:
                return lb
        return out

    def _run_one(self, prompt: str, sample: Sample) -> Prediction:
        text = self.template.format(prompt=prompt, text=sample.text)
        try:
            raw = self.llm.invoke(text).content
        except Exception as e:  # lỗi mạng/quota -> đánh dấu, không làm sập cả run
            return Prediction(sample_id=sample.id, output=f"__ERROR__: {e}",
                              model=self.model)
        return Prediction(sample_id=sample.id, output=self._normalize(raw),
                          model=self.model)

    def execute(self, prompt: str, samples: List[Sample]) -> List[Prediction]:
        if self.num_workers <= 1:
            out = []
            for i, s in enumerate(samples):
                if self.delay and i:      # không nghỉ trước lần gọi đầu
                    time.sleep(self.delay)
                out.append(self._run_one(prompt, s))
            return out
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            return list(ex.map(lambda s: self._run_one(prompt, s), samples))
