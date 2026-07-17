# Hỏi đáp

Đây là những câu mình đoán sẽ bị hỏi, kèm câu trả lời và một lệnh để chứng minh
ngay tại chỗ. Không cần tin lời mình, cứ chạy thử.

Các lệnh chạy từ thư mục gốc repo. Trừ mục 10, tất cả đều không gọi LLM nên không
tốn tiền và xong trong vài giây.

---

## 1. Đây là framework hay chỉ là một ứng dụng?

Framework. Khác biệt nằm ở chỗ ai giữ vòng lặp chính.

Với một ứng dụng, bạn là người gọi nó. Với framework thì ngược lại: nó giữ vòng
lặp và gọi ngược vào code của bạn. Người ta gọi cái đó là Inversion of Control.
Ở đây bạn chỉ cắm component vào bốn điểm, còn vòng lặp không thuộc về bạn.

```bash
grep -n "executor.execute\|evaluator.evaluate\|store.record_score\|optimizer.propose" \
  prompt_tuning_framework/core/tuner.py
```

Lệnh in ra khoảng 10 dòng. Bốn dòng đầu (7–10) là docstring mô tả vòng lặp. Bốn
dòng quan trọng là 102, 104, 121, 146 — chúng nằm bên trong vòng `for` của
framework. Hai dòng cuối là lúc chấm tập test.

Điểm cần thấy: cả bốn lời gọi đó đều nằm trong code của framework, không phải
code của người dùng. Người dùng chỉ đưa component vào rồi gọi `tuner.run()` đúng
một lần.

---

## 2. AutoPrompt đứng ở đâu trong kiến trúc?

Nó là một plugin optimizer, ngang hàng với optimizer mình tự viết. Đổi qua lại
bằng một dòng YAML. Nếu AutoPrompt nằm ở lõi thì cái này đã là ứng dụng bọc quanh
AutoPrompt, chứ không phải framework.

```bash
python -c "
import prompt_tuning_framework.components
from prompt_tuning_framework import available
for k in ('store','executor','evaluator','optimizer'):
    print(f'{k:10}:', available(k))
"
```

Kết quả in ra `optimizer: ['autoprompt', 'llm_rewrite']`. Nó là một phần tử trong
danh sách, không phải trung tâm.

Mà không chỉ đăng ký được — nó chạy thật. Engine AutoPrompt cắm vào qua interface
`BaseOptimizer` đạt 85.0/100 trên tập test, CI 95% [79.4, 89.3] (train 60 / test
200). Chi tiết ở README mục "Kết quả thật". Lõi framework không sửa dòng nào.

Nếu bị vặn "sao plugin chỉ 85 mà optimizer nhà được 100?", trả lời thẳng: hai lần
chạy khác điều kiện, một bên 280 mẫu train một bên 60, nên không so trực tiếp
được. Mình cũng không dùng nó để nói engine nào hơn. Nó chỉ trả lời đúng một câu:
engine ngoài có cắm vào chạy được không.

---

## 3. Prompt tối ưu có thật sự tốt hơn, hay chỉ may?

McNemar ghép cặp trên tập test cho p = 1.4 × 10⁻¹⁷. Cụ thể là 57 ca chuyển từ sai
sang đúng, không ca nào xấu đi.

Cách hiểu con số đó: nếu hai prompt thực sự ngang nhau thì mỗi ca lệch giống như
tung một đồng xu. Ở đây có 57 ca lệch và cả 57 đều nghiêng về prompt mới. Xác
suất tung 57 đồng xu ra cùng một mặt chính là 2/2⁵⁷, tức đúng bằng p-value.

```bash
python -c "
from prompt_tuning_framework import mcnemar_exact, min_flips_for_significance
print('Ket qua that (57 ca lat, 0 ca xau di):', mcnemar_exact(0, 57))
print('Tung 57 dong xu ra cung mot mat  :', 2 / 2**57)
print('Nguong can de dat p<0.05         :', min_flips_for_significance(0.05), 'ca lat')
"
```

Dùng McNemar chứ không so hai khoảng tin cậy rời nhau, vì hai prompt chạy trên
cùng một bộ mẫu. Ca nào cả hai cùng đúng thì không giúp phân biệt được gì.

---

## 4. Điểm 100/100 có đáng tin không?

Điểm trần luôn dễ gây hiểu nhầm, nên mình không bao giờ báo nó trần trụi. Trên
200 ca, 100 điểm cho khoảng [98.1, 100.0], nghĩa là chỉ chứng minh được prompt
đúng ít nhất 98.1%. Cũng 100 điểm đó mà chỉ có 16 ca thì tụt xuống còn ≥ 80.6%.

```bash
python -c "
from prompt_tuning_framework import wilson_interval
for n in (16, 60, 200):
    lo, hi = wilson_interval(n, n)
    print(f'{n:>3}/{n:<3} = 100 diem  ->  that ra chi chung minh duoc >= {lo:.1f}%')
"
```

Nói cách khác, "100 điểm" tự nó không có nghĩa gì nếu không kèm số mẫu.

---

## 5. Sao biết prompt không phải học thuộc bộ mẫu?

Vì có tập test giữ riêng, và optimizer không bao giờ nhìn thấy nó. Optimizer chỉ
được xem các ca sai của tập train. Điểm công bố thì lấy từ tập test. Train 100 mà
test cũng 100, nên không có chuyện học thuộc.

Đây không phải lo xa. Có một lần chạy trên bộ nhỏ cho train 83.3 nhưng test chỉ
66.7. Nếu không tách tập, mình đã báo cáo con số cao hơn sự thật 16.6 điểm mà
không hề biết.

```bash
python -m pytest prompt_tuning_framework/tests/test_train_test_split.py -v -k "never_sees or overfitting"
```

Có test khoá chặt chuyện "optimizer không bao giờ thấy tập test", và test chứng
minh framework tự cảnh báo HỌC THUỘC khi điểm dev vượt test quá 10 điểm.

---

## 6. Bộ mẫu có dễ quá không? Model đoán mò cũng đúng chăng?

Không, vì bộ mẫu thiết kế theo kiểu giai thừa để mọi luật lười đều thất bại. Quy
định ẩn là khách phải vừa trả tiền vừa bị chặn hoàn toàn. Bám vào một dấu hiệu
đơn lẻ chỉ được 83.3 điểm; muốn 100 thì phải hiểu đúng sự kết hợp.

```bash
python -m pytest prompt_tuning_framework/tests/test_dataset.py -v
```

Đáng chú ý nhất là `test_tone_is_useless_for_guessing`. Ticket gào "URGENT!!!"
được rải đều cả Yes lẫn No, đúng 50/50, nên đoán theo giọng điệu là vô dụng.

Bộ mẫu đầu tiên mình làm hỏng đúng chỗ này: 15/15 ticket gào to đều là No. Model
chỉ cần học "gào to thì No" là xong, và lúc đó benchmark đang đo giọng điệu chứ
không đo quy định. Test này sinh ra để lỗi đó không quay lại.

Bộ PII tiếng Việt cũng vậy, xem `test_pii_dataset.py`. Prompt khởi đầu có sẵn chữ
"nhạy cảm" nên bộ mẫu phải rải chữ đó đều 50/50, nếu không thì ta đang đo từ khoá
chứ không đo việc có lộ dữ liệu hay không.

---

## 7. Làm sao chắc chắn phép đo của em đúng?

Đừng tin mình. Phần thống kê do mình tự viết, nên nó được kiểm chứng bằng ba tầng
độc lập, và CI chạy lại mỗi lần push.

```bash
pip install -e "prompt_tuning_framework/[test,verify]"
python -m pytest prompt_tuning_framework/tests/test_metrics_verification.py -v
```

| Tầng | Cách làm | Bắt được lỗi gì |
|---|---|---|
| 1. Đối chiếu | So với `statsmodels`, thư viện do người khác viết | Cài sai công thức |
| 2. Tham chiếu | Khoá cứng giá trị suy tay (`b=0, c=k` thì `p = 2/2ᵏ`) | Cài sai, kể cả khi máy không có statsmodels |
| 3. Mô phỏng | Kiểm tra tính chất thống kê, không kiểm tra công thức | Dùng sai công cụ |

Tầng 3 mạnh nhất vì nó không dựa vào công thức nào cả: khoảng tin cậy 95% thì
phải chứa giá trị thật khoảng 95% số lần, không thì sai. Có cả chiều ngược lại
nữa — McNemar phải bắt được khác biệt thật trên 95% số lần, vì nếu không thì một
hàm luôn trả về `p = 1.0` cũng qua được bài kiểm tra.

Kết quả: Wilson và McNemar khớp statsmodels, lệch 0.000000.

---

## 8. Điểm cao thế, có phải nhờ dùng model xịn không?

Không. Executor dùng model rẻ nhất (`gemini-3.1-flash-lite`), và nó là thứ chạy
1.320 lượt. Model mạnh chỉ dùng cho optimizer, mà optimizer chỉ gọi 1 lần mỗi
vòng, cả run có 3 lần. Nhờ sự bất đối xứng đó mà một lần chạy đầy đủ chỉ tốn
khoảng 1.300 VND.

```bash
grep -n "DEFAULT_MODELS" -A 4 prompt_tuning_framework/llm.py
```

Prompt tối ưu cũng chuyển được sang model khác. Đo trên 3 model, model yếu nhất
vẫn đạt 95.0, và độ chênh giữa các model giảm từ 8.4 xuống 5.0. Tức là nó ít kén
model hơn prompt gốc, dù chỉ được tinh chỉnh trên đúng một model.

---

## 9. Sao chấm đa model bằng min mà không lấy trung bình?

Vì trung bình cho phép model giỏi che lấp model dở. Một prompt đạt 100 trên model
A và 60 trên model B có trung bình 80, nghe thì ổn, nhưng đó không phải prompt
dùng chung được. Lấy min buộc phải sửa cho model yếu nhất.

```bash
python -m pytest prompt_tuning_framework/tests/test_metrics.py -v -k cross_model
```

Trong đó có một test khoá chi tiết dễ bỏ sót: mẫu số của khoảng tin cậy phải là
số mẫu, không phải số_model × số_mẫu. Cộng dồn vào sẽ làm khoảng hẹp đi một cách
giả tạo, vì đó là cùng một bộ mẫu đo lặp lại chứ không phải quan sát độc lập.

---

## 10. Cho xem chạy thật được không?

Được. Bộ đầy đủ tốn khoảng 1.320 lượt gọi nên hơi lâu, còn bản rút gọn thì vài
phút là xong:

```bash
python -m prompt_tuning_framework.examples.hard_example --nho 24 --max-iters 2
```

Bộ đầy đủ, cần API key đã bật billing, khoảng 1.300 VND:

```bash
python -m prompt_tuning_framework.examples.hard_example --delay 0 --workers 8
```

Bài tiếng Việt (phát hiện lộ thông tin khách hàng) nhẹ hơn, 120 ca:

```bash
python -m prompt_tuning_framework.examples.pii_example
```

---

## Số liệu cần nhớ

Bài ticket, 480 ca chia 280 train / 200 test:

| | Train | Test (optimizer chưa từng thấy) |
|---|---|---|
| Prompt gốc | 68.9 | 71.5 — CI [64.9, 77.3] |
| Prompt tối ưu | 100.0 | 100.0 — CI [98.1, 100.0] |

McNemar p = 1.4 × 10⁻¹⁷, 57 ca lật, 0 ca xấu đi.

Bài PII tiếng Việt, 120 ca chia 60/60: test 66.7 → 100.0, CI [94.0, 100.0],
McNemar p = 1.9 × 10⁻⁶.

Tổng: 205 test, chi phí khoảng 1.300 VND một lần chạy đầy đủ.

---

## Nếu bị hỏi điểm yếu

Cứ nói thẳng. Người chấm thường đánh giá cao chuyện mình tự biết giới hạn của
mình hơn là cố che.

Bộ mẫu sinh bằng generator chứ không phải ticket thật. Nó kiểm tra được khả năng
suy ra quy định ẩn, nhưng ngôn ngữ đơn giản hơn ticket ngoài đời.

Bao phủ của Wilson tụt xuống khoảng 92% khi accuracy sát 100%. Đây là đặc tính đã
biết của phương pháp chứ không phải bug — mình chạy cùng mô phỏng đó với
statsmodels và nó tụt y hệt. Riêng kết quả 200/200 vẫn an toàn, vì Wilson cho
[98.1, 100] gần trùng với Clopper-Pearson [98.2, 100].

Mới chỉ thử trên bài phân loại 2 nhãn. Đã đo trên hai miền khác nhau (ticket
tiếng Anh và PII tiếng Việt) nhưng cả hai đều 2 nhãn. Framework không ràng buộc
điều đó, chỉ là mình chưa đo bài nhiều nhãn hay bài sinh văn bản.

Muốn siết non-inferiority xuống margin 3 điểm thì cần hơn 400 ca test. Bộ hiện
tại chỉ đủ cho margin 5 điểm.
