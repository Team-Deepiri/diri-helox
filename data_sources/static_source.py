"""
StaticDataSource: reads JSONL files from disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

from .base import DataSample, DataSource, DataSourceConfig


class StaticDataSource(DataSource):
    """
    Loads DataSamples from one or more JSONL files.

    Config params:
        file_paths (list[str]): paths to JSONL files (supports glob patterns via Path.glob)
        text_field  (str): field name for text content (default: "text")
        label_field (str): field name for numeric label (default: "label")
        max_samples (int | None): cap on samples loaded
    """

    def __init__(self, config: DataSourceConfig) -> None:
        super().__init__(config)
        raw_paths = config.params.get("file_paths", [])
        self._text_field = config.params.get("text_field", "text")
        self._label_field = config.params.get("label_field", "label")
        self._max_samples = config.params.get("max_samples", None)

        # Resolve glob patterns
        self._file_paths: List[Path] = []
        for p in raw_paths:
            path = Path(p)
            if "*" in str(p) or "?" in str(p):
                self._file_paths.extend(sorted(path.parent.glob(path.name)))
            else:
                self._file_paths.append(path)

    def _read_file(self, path: Path) -> List[DataSample]:
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                text = item.get(self._text_field, "")
                label = item.get(self._label_field)
                samples.append(
                    DataSample(
                        text=text,
                        label=label if isinstance(label, int) else None,
                        label_name=item.get("label_name"),
                        metadata={
                            k: v
                            for k, v in item.items()
                            if k not in (self._text_field, self._label_field, "label_name")
                        },
                        source=str(path),
                    )
                )
        return samples

    def load(self) -> List[DataSample]:
        all_samples: List[DataSample] = []
        for path in self._file_paths:
            if not path.exists():
                raise FileNotFoundError(f"StaticDataSource: file not found: {path}")
            all_samples.extend(self._read_file(path))
            if self._max_samples and len(all_samples) >= self._max_samples:
                break
        if self._max_samples:
            all_samples = all_samples[: self._max_samples]
        return all_samples

    def stream(self) -> Iterator[DataSample]:
        count = 0
        for path in self._file_paths:
            if not path.exists():
                raise FileNotFoundError(f"StaticDataSource: file not found: {path}")
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    text = item.get(self._text_field, "")
                    label = item.get(self._label_field)
                    yield DataSample(
                        text=text,
                        label=label if isinstance(label, int) else None,
                        label_name=item.get("label_name"),
                        metadata={
                            k: v
                            for k, v in item.items()
                            if k not in (self._text_field, self._label_field, "label_name")
                        },
                        source=str(path),
                    )
                    count += 1
                    if self._max_samples and count >= self._max_samples:
                        return

    def get_info(self) -> Dict[str, Any]:
        existing = [p for p in self._file_paths if p.exists()]
        missing = [p for p in self._file_paths if not p.exists()]
        return {
            "source_type": "static",
            "name": self.name,
            "files": [str(p) for p in self._file_paths],
            "existing_files": len(existing),
            "missing_files": [str(p) for p in missing],
        }
