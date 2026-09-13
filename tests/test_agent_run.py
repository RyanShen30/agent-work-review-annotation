import sys
import types

from agentic_review_annotation_distilabel.agents import run
from agentic_review_annotation_distilabel.agents.run import load_instance, load_instances, swebench_image


def test_minus_one_loads_entire_benchmark(monkeypatch):
    rows = [{"instance_id": "one"}, {"instance_id": "two"}]
    monkeypatch.setattr(run, "load_benchmark", lambda path: rows)

    assert load_instances("benchmark", -1) == rows


def test_loads_first_swebench_image(tmp_path, monkeypatch):
    parquet = tmp_path / "part.parquet"
    parquet.write_text("fixture", encoding="utf-8")

    row = {
        "instance_id": "astropy__astropy-12907",
        "problem_statement": "fix it",
    }

    class FakeRow:
        def __init__(self, data):
            self._data = data

        def to_dict(self):
            return self._data

    class FakeRows:
        empty = False

        def __init__(self, rows):
            self._rows = rows
            self.iloc = self

        def __getitem__(self, item):
            if isinstance(item, int):
                return FakeRow(self._rows[item])
            if item == "instance_id":
                return [candidate["instance_id"] for candidate in self._rows]
            return FakeRows(
                [
                    candidate
                    for candidate, include in zip(self._rows, item, strict=True)
                    if include
                ]
            )

        def to_dict(self):
            return self._rows[0]

    fake_pandas = types.SimpleNamespace(
        read_parquet=lambda path: FakeRows([row]),
        concat=lambda frames: list(frames)[0],
    )
    monkeypatch.setitem(sys.modules, "pandas", fake_pandas)

    row = load_instance(tmp_path, 0)

    assert row["instance_id"] == "astropy__astropy-12907"
    assert swebench_image(row) == (
        "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
    )
