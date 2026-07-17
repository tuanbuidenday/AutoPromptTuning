"""Sinh bộ mẫu: văn bản có LỘ THÔNG TIN KHÁCH HÀNG không? (Yes/No)

Quy định ẩn — KHÔNG nói cho model biết, nó phải tự suy ra từ các ca sai:

    Yes  <=>  văn bản chứa ĐỊNH DANH CÁ NHÂN CỦA KHÁCH HÀNG
              (số điện thoại cá nhân, email cá nhân, CCCD, số thẻ, địa chỉ nhà)

Thiết kế giai thừa, cốt để MỌI luật lười đều thất bại. Ba tín hiệu bề mặt mà một
model lười sẽ bám vào, và cách vô hiệu hoá từng cái:

  1. Từ khoá "nhạy cảm"/"bảo mật"  -> rải đều 50/50 giữa Yes và No.
     Bộ mẫu ngây thơ hay để những từ này chỉ xuất hiện ở ca Yes; khi đó model chỉ
     cần học "thấy chữ nhạy cảm -> Yes" là đạt điểm cao, và benchmark đang đo TỪ
     KHOÁ chứ không đo việc có lộ dữ liệu hay không.
  2. Có chuỗi số dài -> rải đều. Ca No vẫn có số: mã đơn hàng, mã vận đơn,
     hotline, số tiền, ngày tháng. "Thấy số -> Yes" phải sai một nửa.
  3. Có tên riêng -> rải đều. Nêu tên khách KHÔNG phải là lộ định danh.

Bốn ô No đều là bẫy thật, mỗi ô đánh vào một hiểu nhầm khác nhau:
     N1  bàn VỀ chủ đề dữ liệu nhạy cảm, không nêu gì cụ thể
     N2  mã đơn/vận đơn (chuỗi số dài) + tên khách
     N3  hotline / email CÔNG TY (số + email, nhưng không phải của khách)
     N4  tên khách + nội dung thường

Toàn bộ dữ liệu là BỊA, sinh từ seed cố định -> chạy lại ra y hệt.
"""
import csv
import random
from pathlib import Path

SEED = 20260717
N_MOI_O = 15          # 4 ô Yes + 4 ô No = 120 ca, cân bằng 60/60
N_TEST = 60

HERE = Path(__file__).resolve().parent
OUT = HERE / "pii.csv"

HO = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Vũ", "Đặng", "Bùi", "Đỗ", "Ngô"]
TEN = ["An", "Bình", "Chi", "Dũng", "Hà", "Khánh", "Linh", "Minh", "Ngọc", "Phúc",
       "Quân", "Sơn", "Thảo", "Trang", "Vy"]
DUONG = ["Lê Lợi", "Trần Hưng Đạo", "Nguyễn Huệ", "Hai Bà Trưng", "Lý Thường Kiệt"]
QUAN = ["Quận 1", "Quận 3", "Quận 7", "Cầu Giấy", "Ba Đình"]
PHONG = ["chăm sóc khách hàng", "kinh doanh", "kho vận", "kỹ thuật", "kế toán"]
HOP_THU = ["hotro", "cskh", "khieunai", "lienhe", "phanhoi"]

# Rải đều giữa Yes và No -> P(Yes | có từ này) = 50%, luật theo từ khoá vô dụng.
NHAY_CAM = ["Lưu ý: nội dung nhạy cảm.", "Thông tin bảo mật, xin cẩn trọng.",
            "Dữ liệu khách hàng, cần bảo mật.", "Nhạy cảm — hạn chế chia sẻ."]


def _ten(r):
    return f"{r.choice(HO)} {r.choice(TEN)}"


def _gan_nhay_cam(r, text, co):
    """Nửa số ca của MỌI ô đều gắn từ khoá nhạy cảm — kể cả ô No."""
    return f"{r.choice(NHAY_CAM)} {text}" if co else text


def _sinh(r):
    rows = []

    def them(nhan, ham):
        """Sinh N_MOI_O ca DUY NHẤT cho một ô.

        Không chỉ gọi ham() đúng N lần: mẫu câu ít mà sinh nhiều thì sẽ trùng.
        Bản đầu tiên của file này mắc đúng lỗi đó — 120 ca nhưng chỉ 111 ca duy
        nhất, và vì split chia theo ca chứ không theo văn bản, 6 văn bản nằm ở CẢ
        train lẫn test. 10% tập test đã bị nhìn thấy lúc học, nên điểm test không
        còn là điểm trên dữ liệu lạ. Vòng lặp này bảo đảm mỗi ô đủ ca duy nhất.
        """
        rieng, canh = [], 0
        while len(rieng) < N_MOI_O:
            t = ham(r)
            if t not in rieng:
                rieng.append(t)
            canh += 1
            if canh > N_MOI_O * 200:
                raise RuntimeError(
                    f"Mẫu câu quá ít, không sinh nổi {N_MOI_O} ca duy nhất cho ô "
                    f"nhãn {nhan!r} — thêm biến thể vào template.")
        for i, t in enumerate(rieng):
            rows.append((_gan_nhay_cam(r, t, i % 2 == 0), nhan))

    # ---------- Yes: có định danh cá nhân của khách ----------
    them("Yes", lambda r: f"Khách {_ten(r)} phản ánh chưa nhận hàng, số điện thoại "
                          f"09{r.randint(10**7, 10**8-1)}.")
    them("Yes", lambda r: f"Đã gửi lại hoá đơn cho {_ten(r)} qua email "
                          f"{r.choice(TEN).lower()}{r.randint(80, 99)}@gmail.com.")
    them("Yes", lambda r: f"Xác minh danh tính khách: CCCD "
                          f"{r.randint(10**11, 10**12-1)}.")
    them("Yes", lambda r: f"Giao cho {_ten(r)}, địa chỉ {r.randint(1, 300)} "
                          f"{r.choice(DUONG)}, {r.choice(QUAN)}.")

    # ---------- No: bốn kiểu bẫy ----------
    # N1 — bàn VỀ chủ đề, không nêu gì cụ thể. Mọi mẫu đều có phần thay đổi được
    # (quý, phòng ban, con số) để sinh đủ ca duy nhất.
    them("No", lambda r: r.choice([
        f"Quy trình xử lý dữ liệu khách hàng cần rà soát lại trong quý "
        f"{r.randint(1,4)}/{r.randint(2024, 2026)}.",
        f"Đội {r.choice(PHONG)} phải hoàn tất đào tạo bảo vệ thông tin cá nhân "
        f"trước ngày {r.randint(1,28)}/{r.randint(1,9)}.",
        f"Báo cáo quý {r.randint(1,4)} cho thấy rủi ro rò rỉ dữ liệu tăng "
        f"{r.randint(5, 40)}%.",
        f"Cần mã hoá {r.randint(3, 90)} bảng dữ liệu định danh trước khi lưu trữ.",
    ]))
    # N2 — chuỗi số dài + tên khách, nhưng là mã đơn (không phải định danh)
    them("No", lambda r: f"Đơn hàng {r.randint(10**9, 10**10-1)} của khách "
                         f"{_ten(r)} đã được đóng gói.")
    # N3 — số + email, nhưng của CÔNG TY
    them("No", lambda r: r.choice([
        f"Vui lòng gọi hotline 1900{r.randint(1000, 9999)} để được hỗ trợ.",
        f"Gửi khiếu nại về {r.choice(HOP_THU)}@congty.vn, phản hồi trong "
        f"{r.randint(2, 48)} giờ.",
    ]))
    # N4 — tên khách + nội dung thường, không định danh (một nửa có số: tiền/ngày)
    them("No", lambda r: r.choice([
        f"Khách {_ten(r)} đánh giá {r.randint(4,5)} sao cho lần giao hàng "
        f"ngày {r.randint(1,28)}/{r.randint(1,9)}.",
        f"{_ten(r)} yêu cầu đổi size, tổng đơn {r.randint(200, 900)}.000đ.",
        f"Đã liên hệ {_ten(r)} ngày {r.randint(1,28)}/0{r.randint(1,9)}, khách hài lòng.",
    ]))
    return rows


def main():
    r = random.Random(SEED)
    rows = _sinh(r)
    r.shuffle(rows)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(rows)
    yes = sum(1 for _, n in rows if n == "Yes")
    print(f"Đã ghi {len(rows)} ca vào {OUT}  ({yes} Yes / {len(rows)-yes} No)")


if __name__ == "__main__":
    main()
