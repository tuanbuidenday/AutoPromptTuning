"""Nạp bộ ca test từ file."""
import csv
from pathlib import Path
from typing import List

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
