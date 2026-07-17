# Hỏi & Đáp — kèm lệnh chứng minh tại chỗ

Mỗi mục: câu hỏi → trả lời ngắn (nói được ra miệng) → lệnh chạy để chứng minh.

Chạy từ thư mục gốc repo. Các lệnh đều **không gọi LLM**, không tốn tiền, chạy
trong vài giây — trừ mục 10.

---

## 1. Đây là framework hay chỉ là một ứng dụng?

**Trả lời:** Framework, vì nó theo nguyên lý **Inversion of Control**: framework
giữ vòng lặp chính và *gọi ngược* vào component của người dùng. Ứng dụng thì
ngược lại — người dùng gọi vào nó. Bốn điểm cắm khớp 1:1 với 4 bước bắt buộc.

```bash
grep -n "executor.execute\|evaluator.evaluate\|store.record_score\|optimizer.propose" \
  prompt_tuning_framework/core/tuner.py
```

Bốn dòng này nằm **bên trong vòng `for` của framework** — người dùng không hề gọi
chúng. Đó là định nghĩa của IoC.

---

## 2. AutoPrompt đứng ở đâu trong kiến trúc?

**Trả lời:** Nó chỉ là **một plugin optimizer**, ngang hàng với optimizer tôi tự
viết. Người dùng đổi bằng một dòng YAML. Nếu AutoPrompt là lõi thì đây đã là ứng
dụng bọc AutoPrompt, không phải framework.

```bash
python -c "
import prompt_tuning_framework.components
from prompt_tuning_framework import available
for k in ('store','executor','evaluator','optimizer'):
    print(f'{k:10}:', available(k))
"
```

Kết quả: `optimizer: ['autoprompt', 'llm_rewrite']` — `autoprompt` là **một phần
tử trong danh sách**, không phải trung tâm.

---

## 3. Prompt tối ưu có thật sự tốt hơn, hay chỉ may mắn?

**Trả lời:** Kiểm định ghép cặp McNemar trên tập test cho **p = 1.4 × 10⁻¹⁷**.
Nghĩa là nếu hai prompt thực sự như nhau, xác suất thấy chênh lệch lớn thế này
do ngẫu nhiên là 1 phần 10¹⁷. 57 ca chuyển từ sai sang đúng, **0 ca xấu đi**.

```bash
python -c "
from prompt_tuning_framework import mcnemar_exact, min_flips_for_significance
print('Ket qua that (57 ca lat, 0 ca xau di):', mcnemar_exact(0, 57))
print('Nguong can de dat p<0.05          :', min_flips_for_significance(0.05), 'ca lat')
"
```

Dùng **McNemar ghép cặp** chứ không so hai khoảng tin cậy rời, vì hai prompt chạy
trên **cùng bộ mẫu** — ca nào cả hai cùng đúng thì không mang thông tin phân biệt.

---

## 4. Điểm 100/100 có đáng tin không?

**Trả lời:** Điểm trần là con số gây hiểu nhầm, nên tôi luôn báo kèm khoảng tin
cậy. Trên 200 ca, 100 điểm cho `[98.1, 100.0]` — tức chứng minh được prompt đúng
**ít nhất 98.1%**. Nếu chỉ có 16 ca thì 100 điểm chỉ chứng minh được ≥ 80.6%.

```bash
python -c "
from prompt_tuning_framework import wilson_interval
for n in (16, 48, 200):
    lo, hi = wilson_interval(n, n)
    print(f'{n:>3}/{n:<3} = 100 diem  ->  that ra chi chung minh duoc >= {lo:.1f}%')
"
```

---

## 5. Sao biết prompt không phải "học thuộc" tập mẫu?

**Trả lời:** Vì tôi **tách tập test giữ riêng** — optimizer không bao giờ nhìn
thấy. Nó chỉ được xem ca sai của tập train. Điểm công bố lấy từ tập test. Kết quả
train 100 = test 100, nên không học thuộc.

**Đây không phải lý thuyết** — một lần chạy trên bộ nhỏ đã cho train 83.3 nhưng
test chỉ 66.7. Nếu không tách, tôi đã báo cáo con số cao hơn sự thật 16.6 điểm.

```bash
python -m pytest prompt_tuning_framework/tests/test_train_test_split.py -v -k "never_sees or overfitting"
```

Có test khoá chặt bất biến "optimizer không bao giờ thấy tập test", và test chứng
minh framework tự phát cảnh báo `HỌC THUỘC` khi điểm dev vượt test quá 10 điểm.

---

## 6. Bộ mẫu có dễ quá không? Model chỉ cần đoán mò cũng đúng?

**Trả lời:** Không, vì bộ mẫu theo **thiết kế giai thừa**: mọi luật lười đều thất
bại. Quy định ẩn là "khách **trả tiền** VÀ bị chặn hoàn toàn" — mỗi dấu hiệu đơn
lẻ chỉ đạt 83.3 điểm, chỉ hiểu đúng **sự kết hợp** mới đạt 100.

```bash
python -m pytest prompt_tuning_framework/tests/test_dataset.py -v
```

Quan trọng nhất là `test_tone_is_useless_for_guessing`: ticket gào "URGENT!!!" rải
đều **cả Yes lẫn No** (`P(Yes | gào to) = 50%`). Bộ mẫu đầu tiên tôi làm có
khiếm khuyết đúng chỗ này — 15/15 ticket gào to đều là No, nên model chỉ cần học
"gào to → No" là xong, và benchmark đang đo **giọng điệu** chứ không đo quy định.

---

## 7. Làm sao chắc chắn phép đo của em đúng?

**Trả lời:** Đừng tin tôi. Phần thống kê do tôi tự viết, nên tôi kiểm chứng bằng
**ba tầng độc lập**, và CI chạy lại mỗi lần push.

```bash
pip install -e "prompt_tuning_framework/[test,verify]"
python -m pytest prompt_tuning_framework/tests/test_metrics_verification.py -v
```

| Tầng | Cách | Bắt lỗi gì |
|---|---|---|
| 1. Đối chiếu | So với `statsmodels` (người khác viết) | Cài sai công thức |
| 2. Tham chiếu | Khoá cứng giá trị suy tay (`b=0,c=k ⇒ p=2/2ᵏ`) | Cài sai, kể cả khi không có statsmodels |
| 3. Mô phỏng | Kiểm tra **tính chất**, không kiểm tra công thức | **Dùng sai công cụ** |

Tầng 3 mạnh nhất vì không dựa vào công thức nào: khoảng tin cậy 95% **phải** chứa
giá trị thật ~95% số lần. Có cả chiều ngược lại — McNemar phải **bắt được** khác
biệt thật >95% số lần, nếu không thì hàm luôn trả `p=1.0` cũng qua được.

Kết quả: Wilson và McNemar khớp statsmodels **lệch 0.000000**.

---

## 8. Điểm cao thế, có phải do dùng model xịn không?

**Trả lời:** Không. Executor dùng model **rẻ nhất** (`gemini-3.1-flash-lite`).
Model mạnh chỉ dùng cho optimizer — gọi **1 lần mỗi vòng**, tổng 3 lần cả run.
Chính sự bất đối xứng đó khiến một lần chạy đầy đủ chỉ tốn **~1.300 VND**.

```bash
grep -n "DEFAULT_MODELS" -A 4 prompt_tuning_framework/llm.py
```

Và prompt tối ưu **chuyển được sang model khác**: đo trên 3 model, model yếu nhất
vẫn đạt 95.0, và độ chênh giữa các model **giảm** từ 8.4 xuống 5.0 — tức nó *ít
kén model hơn* prompt gốc, dù chỉ được tinh chỉnh trên một model.

---

## 9. Sao lại chấm đa model bằng min mà không phải trung bình?

**Trả lời:** Vì trung bình cho phép model giỏi **che lấp** model dở. Prompt đạt
100 trên model A và 60 trên model B có trung bình 80 — nghe ổn, nhưng đó không
phải prompt dùng chung được. Lấy min buộc phải sửa cho model yếu nhất.

```bash
python -m pytest prompt_tuning_framework/tests/test_metrics.py -v -k cross_model
```

Có test khoá cả một chi tiết tinh tế: mẫu số của khoảng tin cậy là **số mẫu**,
không phải số_model × số_mẫu — cộng dồn sẽ làm khoảng hẹp đi **giả tạo** vì đó là
cùng bộ mẫu đo lặp, không phải quan sát độc lập.

---

## 10. Cho xem chạy thật được không?

**Trả lời:** Được, nhưng bộ đầy đủ tốn ~1.320 lượt gọi. Bản rút gọn chạy trong
vài phút:

```bash
python -m prompt_tuning_framework.examples.hard_example --nho 24 --max-iters 2
```

Bộ đầy đủ (cần API key đã bật billing, ~1.300 VND):

```bash
python -m prompt_tuning_framework.examples.hard_example --delay 0 --workers 8
```

---

## Số liệu cần nhớ

| | Train (280) | **Test (200, chưa từng thấy)** |
|---|---|---|
| Prompt gốc | 68.9 | **71.5** — CI [64.9, 77.3] |
| Prompt tối ưu | 100.0 | **100.0** — CI [98.1, 100.0] |

**McNemar p = 1.4 × 10⁻¹⁷** · 480 ca (280/200) · 188 test · chi phí ~1.300 VND

---

## Nếu bị hỏi điểm yếu

Trả lời thẳng, đừng giấu — người chấm thường đánh giá cao việc tự biết giới hạn:

- **Bộ mẫu sinh bằng generator**, không phải ticket thật. Nó kiểm tra được việc
  suy ra quy định ẩn, nhưng ngôn ngữ đơn giản hơn ticket thật.
- **Bao phủ của Wilson tụt ~92% khi accuracy sát 100%** — đặc tính đã biết của
  phương pháp (đã kiểm chứng statsmodels tụt y hệt). Kết quả 200/200 vẫn an toàn
  vì Wilson `[98.1, 100]` gần trùng Clopper-Pearson `[98.2, 100]`.
- **Chỉ thử trên bài phân loại 2 nhãn.** Framework không ràng buộc điều đó, nhưng
  tôi chưa đo trên bài khác.
- **Muốn siết non-inferiority xuống margin 3 điểm thì cần hơn 400 ca test**, bộ
  hiện tại chỉ đủ cho 5 điểm.
