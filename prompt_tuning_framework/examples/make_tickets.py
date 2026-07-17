"""Sinh bộ mẫu ticket cho hard_example — thiết kế giai thừa, chống luật proxy.

Chạy:  ./venv/bin/python -m prompt_tuning_framework.examples.make_tickets

Quy định ẩn cần đo (KHÔNG nói cho model biết):
    Yes  <=>  khách ĐANG TRẢ TIỀN  VÀ  bị chặn hoàn toàn
    No   <=>  mọi thứ khác

Vì sao phải sinh theo thiết kế thay vì viết tay tuỳ hứng: một benchmark chỉ có
giá trị khi các LUẬT LƯỜI không ăn được điểm cao. Bộ 96 mẫu viết tay trước đây
mắc đúng lỗi này — 15/15 ticket gào "URGENT!!!" đều là No, nên "gào to -> No" là
luật đúng 100% trong nhóm đó, dù nó chẳng liên quan gì tới quy định thật.

Thiết kế 2×2×2 (giọng điệu × trả tiền × bị chặn) để mỗi dấu hiệu ĐƠN LẺ đều
không đủ kết luận. Mỗi ô chia đôi gào/bình tĩnh:

    trả tiền + bị chặn        -> Yes   (240)
    trả tiền + KHÔNG chặn     -> No    ( 80)
    free     + bị chặn        -> No    ( 80)   <- ca then chốt
    free     + KHÔNG chặn     -> No    ( 80)

Tổng 480 ca: 240 Yes / 240 No, và 240 gào / 240 bình tĩnh.

Vì sao 480 chứ không phải 400: tách 280 train / 200 test. Cần đúng 200 ca test
mới đủ kết luận "rút ngắn prompt mà accuracy không tụt quá 5 điểm" — 120 ca chỉ
kết luận được ở mức 7 điểm. Phần train dư ra không giúp optimizer thông minh hơn
(nó chỉ đọc max_errors=5 ca sai mỗi vòng) nhưng làm điểm dev ổn định hơn và cho
nó một rổ lỗi đa dạng hơn để chọn.

Hệ quả (test_bo_mau.py canh giữ):
    - giọng điệu: P(Yes | gào) ≈ P(Yes | bình tĩnh) ≈ 50%  -> vô dụng để đoán
    - "trả tiền" một mình  -> ~75%, chưa đủ
    - "bị chặn" một mình   -> ~75%, chưa đủ
    - chỉ KẾT HỢP cả hai   -> 100%
"""
import csv
import random
from pathlib import Path

OUT = Path(__file__).parent / "tickets.csv"
SEED = 20260717

# --- chủ thể ĐANG TRẢ TIỀN (nhiều cách diễn đạt, tránh trùng từ khoá) --------
PAYING = [
    "An enterprise customer", "Our largest paying client", "A Pro-plan subscriber",
    "A customer on the Business tier", "A paid account holder", "Our premium customer",
    "A team on an annual contract", "A paying customer", "The client who renewed last month",
    "An account with an active subscription", "A customer billed monthly",
    "Our enterprise tenant", "A paid workspace", "A subscriber on the Growth plan",
    "A customer paying two thousand a month", "One of our contracted accounts",
    "A licensed customer", "A customer on the Scale tier", "Our long-standing paid client",
    "A fully onboarded paying team", "A customer whose invoice cleared last week",
    "A Business-plan organisation", "An account under a signed SLA",
    "A revenue-generating account", "A customer with two hundred paid seats",
    "A commercial customer", "An account on our highest paid tier",
    "A subscribed organisation",
]

# --- chủ thể KHÔNG trả tiền -------------------------------------------------
FREE = [
    "A free-tier user", "Someone on a free trial", "A trial account",
    "An evaluation account", "A prospect on the free plan", "A user who has never paid",
    "A free-plan customer", "Someone still evaluating us", "An unpaid pilot account",
    "A user on the community tier", "A trial workspace", "A sandbox account",
    "A free signup from yesterday", "A user whose trial has not converted",
    "An account with no subscription", "A non-paying user",
    "Someone on our free-forever plan", "A proof-of-concept account",
    "A user in the fourteen-day trial", "An account with zero paid seats",
    "A lead trying the product", "A free-tier developer", "An unlicensed account",
    "A demo account", "A user who declined to upgrade", "A trial org",
    "A free workspace", "An account that never entered billing",
]

# --- BỊ CHẶN HOÀN TOÀN ------------------------------------------------------
BLOCKED = [
    "cannot log in at all", "gets a 500 error on every request",
    "is completely locked out", "sees a blank page everywhere in the product",
    "cannot reach any of their data", "has every API call fail",
    "is stuck on an infinite loading spinner", "cannot perform a single action",
    "has their whole workspace returning 403", "cannot authenticate through any method",
    "gets a timeout on every page", "has all their jobs failing immediately",
    "cannot save anything at all", "has the app crash on launch every time",
    "cannot open any project", "has every write rejected by the server",
    "sees a 502 on every single page", "cannot upload any file",
    "has their entire tenant offline", "cannot access any feature whatsoever",
    "has all exports failing at the first row", "cannot reach the API at all",
    "has their dashboard permanently empty", "gets an error on every click",
    "cannot complete login because the session dies instantly",
    "has every integration call returning an error", "cannot load the workspace at all",
    "has all their data unreachable", "cannot use the product in any way",
    "gets a maintenance page on every route",
]

# --- KHÔNG bị chặn: mỹ thuật / có cách né / câu hỏi / đòi tính năng ---------
NOT_BLOCKED = [
    "reports that the logo is misaligned on the settings page",
    "wants dark mode added", "found a typo in the welcome email",
    "says the button colour does not match their brand",
    "reports the footer year is out of date",
    "asks for a larger font in the sidebar",
    "notes the tooltip is slightly cut off on wide screens",
    "would like the export file renamed",
    "prefers the spinner to rotate the other way",
    "found a broken anchor link in the FAQ",
    "would prefer a square avatar instead of a circle",
    "says the invoice PDF has the address on the wrong line",
    "thinks the dropdown animation is a bit slow",
    "wants the success message to stay five seconds instead of three",
    "spotted an extra space in the page title",
    "reports login fails in Internet Explorer 11 but works fine in Chrome",
    "says the export takes eight seconds but does finish",
    "asks how the billing cycle works",
    "would like a CSV export added next quarter",
    "finds the mobile layout awkward on old tablets, though desktop is fine",
    "asks whether German is on the roadmap",
    "says report generation is slow on huge datasets but completes",
    "requests a training session for their new hires",
    "asks if we integrate with a tool we do not support yet",
    "notes the dark theme contrast is low but still readable",
    "asks for a discount on their renewal",
    "says sync takes two minutes instead of one but finishes correctly",
    "requests an extra filter in the reporting view",
    "asks how to rotate their API key",
    "wants to schedule a call about our roadmap",
    "reports a small rounding difference they can work around",
    "asks for the documentation in PDF form",
    "suggests reordering the navigation menu",
    "would like email digests weekly instead of daily",
]

# --- giọng điệu: PHẢI xuất hiện đều ở cả Yes lẫn No -------------------------
# Nếu chỉ ticket No mới gào to thì "gào to -> No" thành luật đúng tuyệt đối, và
# benchmark sẽ đo giọng điệu thay vì đo quy định.
SHOUT_PRE = [
    "URGENT!!!", "CRITICAL:", "EMERGENCY —", "TOP PRIORITY!!!", "ASAP!!!",
    "SEV1:", "THIS IS A DISASTER —", "HIGHEST PRIORITY!", "MAJOR INCIDENT!!!",
    "PLEASE HELP URGENTLY —", "BLOCKER!!!", "ESCALATE NOW:",
]
SHOUT_POST = [
    " Please fix this immediately!!!", " We need this resolved NOW.",
    " This cannot wait!!!", " I want an answer within the hour.",
    " Escalate to your manager please!!!", "",
]
CALM_PRE = ["", "Hi team,", "Hello,", "Quick note:", "FYI:", "Morning —"]
CALM_POST = ["", " Please advise.", " Thanks.", " Let us know when you can.",
             " No rush on our side.", " Happy to give more detail."]


def _cau(rng, pre, subj, pred, post):
    subj = subj if not pre else subj[0].lower() + subj[1:]
    thanh_phan = [p for p in (pre, f"{subj} {pred}.".strip(), post.strip()) if p]
    return " ".join(thanh_phan)


def _sinh(rng, subjects, predicates, gao, n):
    """Lấy n tổ hợp KHÁC NHAU từ subjects × predicates."""
    combos = [(s, p) for s in subjects for p in predicates]
    if n > len(combos):
        raise ValueError(f"Cần {n} mẫu nhưng chỉ có {len(combos)} tổ hợp.")
    rng.shuffle(combos)
    pre_pool, post_pool = (SHOUT_PRE, SHOUT_POST) if gao else (CALM_PRE, CALM_POST)
    return [_cau(rng, rng.choice(pre_pool), s, p, rng.choice(post_pool))
            for s, p in combos[:n]]


def build():
    rng = random.Random(SEED)
    rows = []

    # Yes: trả tiền VÀ bị chặn — một nửa gào, một nửa bình tĩnh
    for gao in (True, False):
        rows += [(t, "Yes") for t in _sinh(rng, PAYING, BLOCKED, gao, 120)]

    # No: trả tiền nhưng KHÔNG bị chặn
    for gao in (True, False):
        rows += [(t, "No") for t in _sinh(rng, PAYING, NOT_BLOCKED, gao, 40)]

    # No: bị chặn hoàn toàn nhưng KHÔNG trả tiền (ca then chốt)
    for gao in (True, False):
        rows += [(t, "No") for t in _sinh(rng, FREE, BLOCKED, gao, 40)]

    # No: không trả tiền, cũng không bị chặn
    for gao in (True, False):
        rows += [(t, "No") for t in _sinh(rng, FREE, NOT_BLOCKED, gao, 40)]

    rng.shuffle(rows)

    trung = len(rows) - len({t for t, _ in rows})
    if trung:
        raise AssertionError(f"Có {trung} ticket trùng nhau — pool chưa đủ đa dạng.")
    return rows


def main():
    rows = build()
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(rows)

    n_yes = sum(1 for _, l in rows if l == "Yes")
    print(f"Đã ghi {len(rows)} ticket vào {OUT}")
    print(f"  Yes: {n_yes}   No: {len(rows) - n_yes}")


if __name__ == "__main__":
    main()
