"""UI demo — một CONSUMER của Prompt Tuning Framework.

UI này không chứa logic tối ưu: nó chỉ dựng component và gọi PromptTuner.
Toàn bộ vòng lặp do framework điều khiển (IoC).

Chạy:  ./venv/bin/streamlit run prompt_tuning_framework/ui/streamlit_app.py
"""
import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prompt_tuning_framework import (BaseCallback, PromptTuner,  # noqa: E402
                                     Sample, available, create)
from prompt_tuning_framework.components import AccuracyEvaluator  # noqa: E402
from prompt_tuning_framework.components import (InMemoryPromptStore,
                                                SQLitePromptStore)
from prompt_tuning_framework.llm import default_model  # noqa: E402

st.set_page_config(page_title="Prompt Tuning Framework", page_icon="🧩", layout="wide")

DEFAULT_CSV = """text,label
It barks loudly whenever a stranger walks past the gate.,Yes
It purrs on my lap and kneads the blanket.,No
Loyal companion that fetches the ball every morning.,Yes
It climbs the bookshelf and refuses to come down.,No
We take it for a walk on a leash twice a day.,Yes
It scratches the post and grooms itself for hours.,No
A furry friend that wags its tail when I come home.,Yes
Independent pet that meows at 5am demanding food.,No
"""


class StreamlitProgress(BaseCallback):
    """Hook: framework gọi lại sau mỗi vòng để cập nhật UI."""

    def __init__(self, box, bar, max_iters):
        self.box, self.bar, self.max_iters = box, bar, max_iters
        self.rows = []

    def on_iteration_end(self, iteration, version, result):
        self.rows.append({
            "Vòng": iteration,
            "Điểm /100": result.score,
            "Đúng": result.num_correct,
            "Sai": len(result.errors),
        })
        self.box.dataframe(self.rows, use_container_width=True, hide_index=True)
        self.bar.progress(min((iteration + 1) / self.max_iters, 1.0))


st.title("🧩 Prompt Tuning Framework")
st.caption("UI demo — một *consumer* của framework. Vòng lặp do `PromptTuner` điều khiển.")

# ---------------- Sidebar: dựng component từ registry ----------------
with st.sidebar:
    st.header("⚙️ Cấu hình")

    labels_raw = st.text_input("Nhãn (cách nhau bởi dấu phẩy)", value="Yes, No")
    labels = [x.strip() for x in labels_raw.split(",") if x.strip()]

    st.markdown("**Plugin** (đọc từ registry của framework)")
    exec_name = st.selectbox("Executor", available("executor"),
                             index=available("executor").index("llm"))
    opt_name = st.selectbox("Optimizer", available("optimizer"),
                            help="Đổi engine tối ưu — AutoPrompt chỉ là 1 plugin")
    provider = st.selectbox(
        "Provider", ["google", "openai"],
        help="Key đọc từ biến môi trường GOOGLE_API_KEY / OPENAI_API_KEY")
    # Model mặc định bám theo provider (xem llm.DEFAULT_MODELS) — không ghim cứng
    # tên model của một hãng, tránh gửi tên model Gemini sang OpenAI.
    exec_model = st.text_input("Model chạy prompt",
                               default_model(provider, "executor"))
    opt_model = st.text_input("Model tối ưu prompt",
                              default_model(provider, "optimizer"))
    # Free tier Gemini chỉ cho 15 request/phút. Vượt là ca lỗi, mà ca lỗi bị loại
    # khỏi mẫu số của điểm -> điểm bị thổi phồng.
    delay = st.number_input("Nghỉ giữa 2 request (giây)", 0.0, 30.0, 4.5, 0.5,
                            help="Free tier Gemini = 15 request/phút. Để 0 nếu tài khoản trả phí.")

    store_name = st.selectbox("Store (Quản lý Prompt)", available("store"))
    max_iters = st.slider("Số vòng tối đa", 1, 10, 3)
    target = st.slider("Dừng khi đạt điểm", 50, 100, 100)

# ---------------- Main ----------------
tab_run, tab_arch = st.tabs(["🚀 Tối ưu prompt", "🧠 Kiến trúc"])

with tab_run:
    c1, c2 = st.columns(2)
    with c1:
        task = st.text_area(
            "Mô tả task",
            "Classify whether a text describes a dog or a cat. "
            "Answer Yes for a dog, No for a cat.", height=90)
        prompt = st.text_area("Prompt ban đầu (cần tối ưu)",
                              "Is this a dog? Yes or No", height=90)
    with c2:
        up = st.file_uploader("Dataset CSV (cột text,label)", type="csv")
        csv_text = st.text_area("…hoặc dán CSV vào đây", DEFAULT_CSV, height=190)

    if st.button("🚀 Chạy tối ưu", type="primary", use_container_width=True):
        try:
            df = pd.read_csv(up) if up else pd.read_csv(io.StringIO(csv_text))
            label_col = "label" if "label" in df.columns else "annotation"
            samples = [Sample(id=i, text=str(r["text"]), label=str(r[label_col]))
                       for i, r in df.iterrows()]
        except Exception as e:
            st.error(f"Không đọc được dataset: {e}")
            st.stop()

        # Dựng component qua registry — UI không biết chi tiết lớp nào
        executor = create("executor", exec_name, provider=provider,
                          model=exec_model, labels=labels, delay=delay)
        optimizer = create("optimizer", opt_name, provider=provider,
                           model=opt_model, labels=labels)
        store = (SQLitePromptStore(db_path="prompt_versions.db", run_name="ui")
                 if store_name == "sqlite" else InMemoryPromptStore())

        st.markdown("#### Tiến trình")
        prog_box, bar = st.empty(), st.progress(0.0)
        tuner = PromptTuner(
            executor=executor, evaluator=AccuracyEvaluator(), optimizer=optimizer,
            store=store, task_description=task, max_iters=max_iters,
            target_score=float(target),
            callbacks=[StreamlitProgress(prog_box, bar, max_iters)],
        )

        with st.spinner("Framework đang chạy vòng lặp…"):
            try:
                best = tuner.run(prompt, samples)
            except Exception as e:
                st.error(f"Chạy lỗi: {e}")
                st.stop()

        history = tuner.store.history()
        first = history[0]

        st.markdown("#### Kết quả")
        m1, m2, m3 = st.columns(3)
        m1.metric("Điểm ban đầu", f"{first.score}/100")
        m2.metric("Điểm tốt nhất", f"{best.score}/100",
                  delta=f"+{round(best.score - first.score, 1)}")
        m3.metric("Số phiên bản", len(history))

        a, b = st.columns(2)
        a.markdown("**📥 Prompt ban đầu**"); a.code(first.text)
        b.markdown("**📤 Prompt đã tối ưu**"); b.code(best.text)

        scored = [h for h in history if h.score is not None]
        if len(scored) > 1:
            st.markdown("#### Điểm qua từng vòng")
            st.line_chart({"Điểm /100": [h.score for h in scored]})

        st.markdown("#### Các phiên bản prompt")
        for v in history:
            star = " 🏆" if v.version == best.version else ""
            with st.expander(f"v{v.version} · {v.score}/100{star}"):
                st.code(v.text)

        st.markdown("#### Đánh giá đúng / sai từng ca")
        preds = tuner.executor.execute(best.text, samples)
        res = AccuracyEvaluator().evaluate(best.text, preds, samples)
        st.dataframe(
            [{"Ca test": r.sample.text, "Đoán": r.predicted, "Đáp án": r.expected,
              "Kết quả": "✅ Đúng" if r.correct else ("⚠️ Bỏ qua" if r.correct is None else "❌ Sai")}
             for r in res.results],
            use_container_width=True, hide_index=True,
        )

with tab_arch:
    st.markdown("""
### Vòng lặp khép kín — framework giữ quyền điều khiển (IoC)

```
PromptTuner.run()          ← framework nắm vòng lặp
   │  mỗi vòng gọi ngược lại component của bạn:
   ├─ ② executor.execute(prompt, samples)   → Thực thi
   ├─ ③ evaluator.evaluate(...)             → Đánh giá
   ├─ ① store.record_score(...)             → Quản lý Prompt
   └─ ④ optimizer.propose(errors)           → Tối ưu hóa
```

**UI này không chứa logic tối ưu** — nó chỉ dựng component rồi gọi `PromptTuner`.
Đó là bằng chứng framework dùng được bởi ứng dụng bên ngoài.
    """)
    st.markdown("#### Plugin đang đăng ký trong registry")
    st.dataframe(
        [{"Loại": k, "Plugin": ", ".join(available(k))}
         for k in ("store", "executor", "evaluator", "optimizer")],
        use_container_width=True, hide_index=True,
    )
