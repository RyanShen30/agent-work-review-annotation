from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetAdapter:
    name: str
    cwd: str
    image_field: str = "image"
    image_repository: str | None = None

    def image(self, instance: dict[str, Any]) -> str:
        value = instance.get(self.image_field)
        if isinstance(value, str) and value.strip():
            value = value.strip()
            return f"{self.image_repository}:{value}" if self.image_repository else value
        if self.image_repository:
            raise ValueError(
                f"{instance.get('instance_id', '<unknown>')} is missing "
                f"{self.image_field}"
            )
        return swebench_image(instance)


ADAPTERS = {
    "swebench_verified": DatasetAdapter("swebench_verified", "/testbed"),
    "swebench_multilingual": DatasetAdapter("swebench_multilingual", "/testbed"),
    "swebench_pro": DatasetAdapter(
        "swebench_pro",
        "/app",
        image_field="dockerhub_tag",
        image_repository="jefzda/sweap-images",
    ),
}


def get_dataset_adapter(
    name: str = "auto",
    benchmark_path: Path | str | None = None,
    instance: dict[str, Any] | None = None,
) -> DatasetAdapter:
    if name != "auto":
        try:
            return ADAPTERS[name]
        except KeyError as exc:
            raise ValueError(f"unsupported generation dataset: {name}") from exc

    if instance and instance.get("dockerhub_tag"):
        return ADAPTERS["swebench_pro"]
    path = str(benchmark_path or "").lower().replace("-", "_")
    if "swe_bench_multilingual" in path:
        return ADAPTERS["swebench_multilingual"]
    if "swe_bench_pro" in path:
        return ADAPTERS["swebench_pro"]
    return ADAPTERS["swebench_verified"]


def load_instances(path: Path | str, selector: int | str) -> list[dict[str, Any]]:
    if str(selector) == "-1":
        return load_benchmark(path)
    return [load_instance(path, selector)]


def load_benchmark(path: Path | str) -> list[dict[str, Any]]:
    rows = _read_parquet(path)
    return [rows.iloc[index].to_dict() for index in range(len(rows))]


def load_instance(path: Path | str, selector: int | str) -> dict[str, Any]:
    rows = _read_parquet(path)
    if isinstance(selector, int) or str(selector).isdigit():
        return rows.iloc[int(selector)].to_dict()
    matches = rows[rows["instance_id"] == selector]
    if matches.empty:
        raise KeyError(f"benchmark instance not found: {selector}")
    return matches.iloc[0].to_dict()


def swebench_image(instance: dict[str, Any]) -> str:
    instance_id = str(instance["instance_id"]).replace("__", "_1776_").lower()
    return f"swebench/sweb.eval.x86_64.{instance_id}:latest"


def _read_parquet(path: Path | str) -> Any:
    import pandas as pd

    path = Path(path)
    files = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {path}")
    return pd.concat(pd.read_parquet(file) for file in files)
