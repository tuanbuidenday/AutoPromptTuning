"""Test ① Quản lý Prompt — cả 2 store phải hành xử giống nhau."""
import pytest

from prompt_tuning_framework.components import InMemoryPromptStore, SQLitePromptStore


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryPromptStore()
    return SQLitePromptStore(db_path=str(tmp_path / "t.db"), run_name="test")


def test_save_auto_increments_version(store):
    v0 = store.save("prompt A")
    v1 = store.save("prompt B")
    assert (v0.version, v1.version) == (0, 1)
    assert v0.score is None  # chưa chấm


def test_record_score_and_history(store):
    v0 = store.save("A")
    v1 = store.save("B")
    store.record_score(v0.version, 40.0)
    store.record_score(v1.version, 90.0)

    hist = store.history()
    assert [h.version for h in hist] == [0, 1]
    assert [h.score for h in hist] == [40.0, 90.0]
    assert [h.text for h in hist] == ["A", "B"]


def test_best_returns_highest_score(store):
    store.save("A"); store.record_score(0, 40.0)
    store.save("B"); store.record_score(1, 90.0)
    store.save("C"); store.record_score(2, 70.0)

    best = store.best()
    assert best.version == 1 and best.score == 90.0 and best.text == "B"


def test_best_is_none_before_any_scoring(store):
    store.save("A")
    assert store.best() is None


def test_metadata_is_preserved(store):
    store.save("A", {"source": "initial"})
    assert store.history()[0].metadata["source"] == "initial"


def test_sqlite_persists_across_reopen(tmp_path):
    """Registry SQLite phải đọc lại được sau khi tạo instance mới."""
    db = str(tmp_path / "persist.db")
    s1 = SQLitePromptStore(db_path=db, run_name="run1")
    s1.save("prompt cu"); s1.record_score(0, 55.0)

    s2 = SQLitePromptStore(db_path=db, run_name="run1")
    assert [h.text for h in s2.history()] == ["prompt cu"]
    assert s2.best().score == 55.0


def test_sqlite_isolates_by_run_name(tmp_path):
    db = str(tmp_path / "multi.db")
    a = SQLitePromptStore(db_path=db, run_name="A")
    b = SQLitePromptStore(db_path=db, run_name="B")
    a.save("cua A")
    b.save("cua B")
    assert [h.text for h in a.history()] == ["cua A"]
    assert [h.text for h in b.history()] == ["cua B"]
