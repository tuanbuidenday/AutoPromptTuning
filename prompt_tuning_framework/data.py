"""Nạp bộ ca test từ file."""
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

from .core.types import Sample


def load_samples_csv(path: str, text_col: str = "text",
                     label_col: str = "label") -> List[Sample]:
    """Đọc CSV có cột text + label (nhãn đúng).

    Chấp nhận cả cột tên 'annotation' thay cho 'label' (định dạng của AutoPrompt).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Không thấy file dataset: {path}")

    samples: List[Sample] = []
    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"File rỗng hoặc thiếu header: {path}")
        if label_col not in reader.fieldnames and "annotation" in reader.fieldnames:
            label_col = "annotation"
        if text_col not in reader.fieldnames:
            raise ValueError(
                f"Thiếu cột {text_col!r}. Cột đang có: {reader.fieldnames}"
            )
        for i, row in enumerate(reader):
            text = (row.get(text_col) or "").strip()
            if not text:
                continue
            label = (row.get(label_col) or "").strip() or None
            samples.append(Sample(id=i, text=text, label=label))

    if not samples:
        raise ValueError(f"Không đọc được ca test nào từ {path}")
    return samples


def split_samples(samples: List[Sample], test_ratio: float = 0.5,
                  seed: Optional[int] = 0,
                  stratify: bool = True) -> Tuple[List[Sample], List[Sample]]:
    """Chia bộ mẫu thành (dev, test).

    Vì sao bắt buộc phải chia: optimizer được xem các ca SAI để viết lại prompt.
    Nếu chấm điểm trên chính những ca đó thì prompt chỉ đang vá thuộc lòng từng
    ca, và điểm 100/100 thu được là điểm học thuộc — không nói lên điều gì về ca
    mới. Tập test phải được giữ riêng, optimizer không bao giờ nhìn thấy.

    :param test_ratio: tỉ lệ dành cho tập test
    :param seed: cố định để chia lại y hệt; None = ngẫu nhiên mỗi lần
    :param stratify: giữ tỉ lệ nhãn ở hai tập bằng nhau. Cần thiết với bộ mẫu
        nhỏ và lệch nhãn — chia ngẫu nhiên thuần có thể dồn gần hết nhãn hiếm
        về một bên, khiến điểm hai tập không so được với nhau.
    """
    if not samples:
        raise ValueError("Không có mẫu nào để chia.")
    if not 0 < test_ratio < 1:
        raise ValueError(f"test_ratio phải nằm trong (0, 1), nhận {test_ratio}")

    rng = random.Random(seed)

    def _take(items: List[Sample]) -> Tuple[List[Sample], List[Sample]]:
        pool = list(items)
        rng.shuffle(pool)
        n_test = round(len(pool) * test_ratio)
        # Đừng để nhóm nào mất sạch: mỗi bên giữ ít nhất 1 ca nếu có từ 2 ca trở lên.
        if len(pool) >= 2:
            n_test = min(max(n_test, 1), len(pool) - 1)
        return pool[n_test:], pool[:n_test]

    if not stratify or any(s.label is None for s in samples):
        return _take(samples)

    by_label = defaultdict(list)
    for s in samples:
        by_label[s.label].append(s)

    dev: List[Sample] = []
    test: List[Sample] = []
    # Nhãn chỉ có ĐÚNG 1 ca thì không chia đôi được, mà dồn hết một phía sẽ làm
    # phía kia rỗng. Rải luân phiên để hai tập vẫn cân nhất có thể.
    le_ve_dev = True
    for label in sorted(by_label):
        nhom = by_label[label]
        if len(nhom) == 1:
            (dev if le_ve_dev else test).extend(nhom)
            le_ve_dev = not le_ve_dev
            continue
        d, t = _take(nhom)
        dev.extend(d)
        test.extend(t)

    rng.shuffle(dev)
    rng.shuffle(test)
    return dev, test
