"""
SyntheticDataSource: generates synthetic training samples using the existing
generate_synthetic_dataset logic from scripts/generate_synthetic_data.py.
Does NOT duplicate code — imports the generator directly.
"""

from __future__ import annotations

import tempfile
from typing import Any, Dict, Iterator, List, Optional

from .base import DataSample, DataSource, DataSourceConfig
from .synthetic_generator import generate_synthetic_dataset


class SyntheticDataSource(DataSource):
    """
    Generates synthetic task classification samples on-the-fly.

    Config params:
        total_examples (int): number of samples to generate (default: 100)
        use_ollama     (bool): use Ollama for enhanced generation (default: False)
        categories     (list[str] | None): limit to these categories (default: all 31)
    """

    def __init__(self, config: DataSourceConfig) -> None:
        super().__init__(config)
        self._total = config.params.get("total_examples", 100)
        self._use_ollama = config.params.get("use_ollama", False)
        self._categories: Optional[List[str]] = config.params.get("categories", None)
        self._cached: Optional[List[DataSample]] = None

    def _generate(self) -> List[DataSample]:
        if self._cached is not None:
            return self._cached

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = generate_synthetic_dataset(
                total_examples=self._total,
                output_dir=tmp_dir,
                use_ollama=self._use_ollama,
            )

        samples: List[DataSample] = []
        for split in ("train", "val", "test"):
            for item in result.get(split, []):
                label_id = item.get("label_id")
                label_name = item.get("label", "")

                # Filter by categories if specified
                if self._categories and label_name not in self._categories:
                    continue

                samples.append(
                    DataSample(
                        text=item["text"],
                        label=label_id,
                        label_name=label_name,
                        metadata={"split": split},
                        source=f"synthetic:{self.name}",
                    )
                )

        self._cached = samples
        return samples

    def load(self) -> List[DataSample]:
        return self._generate()

    def stream(self) -> Iterator[DataSample]:
        yield from self._generate()

    def get_info(self) -> Dict[str, Any]:
        return {
            "source_type": "synthetic",
            "name": self.name,
            "total_examples": self._total,
            "use_ollama": self._use_ollama,
            "categories_filter": self._categories,
        }
