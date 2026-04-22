"""
Importable synthetic dataset generator.

This module contains the core generation logic that was previously implemented in
`scripts/generate_synthetic_data.py`. Keeping it importable avoids runtime
`sys.path` mutations and allows `SyntheticDataSource` to call the generator
directly without lint violations (ruff E402).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

# Import semantic analyzer for dynamic analysis
try:
    from utils.semantic_analyzer import get_semantic_analyzer

    HAS_SEMANTIC_ANALYZER = True
except ImportError:
    HAS_SEMANTIC_ANALYZER = False


# Label mapping (31 categories)
LABEL_MAPPING = {
    # Coding breakdown (6 categories)
    "debugging": 0,
    "refactoring": 1,
    "writing_code": 2,
    "programming": 3,
    "running_code": 4,
    "inspecting": 5,
    # Core categories
    "writing": 6,
    "learning_research": 7,
    "learning_study": 8,
    "learning_training": 9,
    "learning_practice": 10,
    "creative": 11,
    "administrative": 12,
    # Team organization (3 categories)
    "team_organization": 13,
    "team_collaboration": 14,
    "team_planning": 15,
    # Computer/Desk-Work
    "research": 16,
    "planning": 17,
    "communication": 18,
    "big_data_analytics": 19,
    "data_processing": 20,
    "design": 21,
    "qa": 22,
    "testing": 23,
    "validation": 24,
    "reporting": 25,
    "documentation": 26,
    "system_admin": 27,
    # New categories
    "ux_ui": 28,
    "security": 29,
    "data_privacy": 30,
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAPPING.items()}

# Common bare verbs that should not appear at the end of a sentence
BARE_VERBS = {
    "write",
    "create",
    "implement",
    "generate",
    "process",
    "review",
    "run",
    "test",
    "validate",
    "inspect",
    "organize",
    "schedule",
    "plan",
    "design",
    "debug",
    "fix",
    "troubleshoot",
    "develop",
    "build",
    "deploy",
    "configure",
    "setup",
    "install",
    "update",
    "analyze",
    "evaluate",
    "assess",
    "monitor",
    "track",
    "measure",
    "prepare",
    "collect",
    "gather",
    "compile",
    "document",
    "refactor",
}


def fix_bare_verb_at_end(text: str) -> str:
    if not text or not text.strip():
        return text

    text = text.strip()
    words = text.split()
    if len(words) < 2:
        return text

    last_word = words[-1].rstrip(".,!?;:").lower()
    if last_word not in BARE_VERBS:
        return text

    verb = words[-1].rstrip(".,!?;:")
    punctuation = "".join(c for c in words[-1] if c in ".,!?;:")
    object_phrase = " ".join(words[:-1])

    object_phrase_lower = object_phrase.lower()
    if object_phrase_lower.startswith("a "):
        object_phrase = object_phrase_lower[2:]
    elif object_phrase_lower.startswith("an "):
        object_phrase = object_phrase_lower[3:]
    elif object_phrase_lower.startswith("the "):
        object_phrase = object_phrase_lower[4:]
    else:
        object_phrase = object_phrase_lower

    verb_capitalized = verb.capitalize()
    fixed_text = f"{verb_capitalized} {object_phrase}{punctuation}".strip()

    object_words = object_phrase.split()
    if object_words and object_words[0] == verb.lower():
        return object_phrase.capitalize() + punctuation

    return fixed_text


def is_valid_sentence(text: str) -> bool:
    if not text or not text.strip():
        return False

    words = text.strip().rstrip(".!?").split()
    if not words:
        return False

    last_word = words[-1].lower().strip(".,!?;:")
    return last_word not in BARE_VERBS


# Task templates for each category
TASK_TEMPLATES: Dict[str, List[str]] = {}


def _load_templates_from_script() -> None:
    """
    Backwards compatible: reuse the existing large TASK_TEMPLATES dictionary
    from the script without requiring callers to import the script directly.
    """
    global TASK_TEMPLATES
    if TASK_TEMPLATES:
        return

    # Importing the script is safe here because we no longer need sys.path hacks;
    # it lives in the same repo and is importable as a module only if scripts is a package.
    # As a fallback, we duplicate only the dictionary by reading it from the file system
    # would be worse, so we require scripts to be importable when this is used.
    from scripts.generate_synthetic_data import TASK_TEMPLATES as _T  # type: ignore

    TASK_TEMPLATES = _T


def generate_variations(
    base_text: str,
    category: str,
    num_variations: int = 3,
    use_ollama: bool = False,
    semantic_analyzer=None,
) -> List[str]:
    variations = [base_text]

    if use_ollama and semantic_analyzer and HAS_SEMANTIC_ANALYZER:
        try:
            prefixes = semantic_analyzer.generate_semantic_prefixes(base_text, category)
            suffixes = semantic_analyzer.generate_semantic_suffixes(base_text, category)

            template_variations = []
            for prefix in prefixes[: min(3, num_variations - 1)]:
                for suffix in suffixes[:1]:
                    variation = f"{prefix} {base_text.lower()}{suffix}".strip()
                    if (
                        variation != base_text
                        and variation not in template_variations
                        and is_valid_sentence(variation)
                    ):
                        template_variations.append(variation)

            variations.extend(template_variations[: num_variations - 1])

            if len(variations) < num_variations:
                try:
                    paraphrases = semantic_analyzer.generate_paraphrases(
                        base_text, category, min(2, num_variations - len(variations))
                    )
                    for paraphrase in paraphrases:
                        if is_valid_sentence(paraphrase):
                            variations.append(paraphrase)
                except Exception:
                    pass

            if len(variations) < num_variations:
                semantic_verbs = semantic_analyzer.extract_semantic_verbs(base_text, category)
                if semantic_verbs:
                    words = base_text.split()
                    if len(words) > 1:
                        for verb in semantic_verbs[: min(3, num_variations - len(variations))]:
                            if verb.lower() != words[0].lower():
                                new_text = f"{verb} {' '.join(words[1:])}"
                                if new_text not in variations and is_valid_sentence(new_text):
                                    variations.append(new_text)
                                    if len(variations) >= num_variations:
                                        break

        except Exception:
            use_ollama = False

    if not use_ollama or len(variations) < num_variations:
        default_prefixes = [
            "I need to",
            "Can you help me",
            "Please",
            "I want to",
            "Help me",
            "I should",
            "Let me",
            "I'm going to",
        ]

        default_suffixes = [
            "",
            " today",
            " this week",
            " as soon as possible",
            " when you have time",
            " - urgent",
            " - important",
        ]

        template_variations = []
        for prefix in default_prefixes[: min(3, num_variations - len(variations))]:
            for suffix in default_suffixes[:1]:
                variation = f"{prefix} {base_text.lower()}{suffix}".strip()
                if (
                    variation != base_text
                    and variation not in variations
                    and is_valid_sentence(variation)
                ):
                    template_variations.append(variation)

        variations.extend(template_variations)

    attempts = 0
    max_attempts = 10
    while len(variations) < num_variations and attempts < max_attempts:
        attempts += 1
        words = base_text.split()

        if len(words) >= 2:
            modifiers = ["Quickly", "Carefully", "Thoroughly", "Efficiently"]
            for modifier in modifiers:
                if words[0][0].isupper():
                    new_text = f"{modifier} {base_text.lower()}".strip()
                else:
                    new_text = f"{modifier} {base_text}".strip()

                if new_text not in variations and is_valid_sentence(new_text):
                    variations.append(new_text)
                    if len(variations) >= num_variations:
                        break

            if len(variations) >= num_variations:
                break

        break

    while len(variations) < num_variations:
        variations.append(base_text)

    return variations[:num_variations]


class AbilityGenerator:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, num_examples: int = 50):
        base_items = [
            {"prompt": "Solve 12 + 37", "target": "49"},
            {"prompt": "Rewrite politely: 'Fix this now.'", "target": "Could you please fix this?"},
            {"prompt": "What is the capital of Japan?", "target": "Tokyo"},
        ]

        items = (base_items * (num_examples // len(base_items) + 1))[:num_examples]
        out_file = self.output_dir / "ability_dataset.jsonl"
        with open(out_file, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        return items


def generate_synthetic_dataset(
    total_examples: int = 7000,
    examples_per_class: int | None = None,
    output_dir: str = str(Path(__file__).parent.parent / "data"),
    use_ollama: bool = False,
    data_type: str = "classification",
) -> Dict:
    output_dir = str(Path(output_dir).resolve())

    if data_type == "ability":
        generator = AbilityGenerator(output_dir)
        ability_data = generator.generate(num_examples=total_examples)
        return {
            "type": "ability",
            "count": len(ability_data),
            "file": str(Path(output_dir) / "ability_dataset.jsonl"),
        }

    _load_templates_from_script()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if examples_per_class:
        total_examples = examples_per_class * len(LABEL_MAPPING)

    examples_per_class = total_examples // len(LABEL_MAPPING)
    remainder = total_examples % len(LABEL_MAPPING)

    semantic_analyzer = None
    if use_ollama and HAS_SEMANTIC_ANALYZER:
        try:
            semantic_analyzer = get_semantic_analyzer()
            if not (semantic_analyzer and semantic_analyzer.check_ollama_available()):
                semantic_analyzer = None
        except Exception:
            semantic_analyzer = None

    all_data = []
    label_counts = Counter()

    for cat_idx, (category, label_id) in enumerate(LABEL_MAPPING.items(), 1):
        templates = TASK_TEMPLATES[category]
        num_examples = examples_per_class + (1 if label_id < remainder else 0)

        variations_per_template = max(1, num_examples // len(templates))
        extra_variations = num_examples % len(templates)

        category_data = []
        if use_ollama and semantic_analyzer is not None and templates:
            try:
                sample_template = templates[0]
                _ = semantic_analyzer.generate_semantic_prefixes(sample_template, category)
                _ = semantic_analyzer.generate_semantic_suffixes(sample_template, category)
                _ = semantic_analyzer.extract_semantic_verbs(sample_template, category)
            except Exception:
                pass

        for i, template in enumerate(templates):
            num_variations = variations_per_template + (1 if i < extra_variations else 0)
            use_semantic_for_this = use_ollama and (i % 20 == 0) and semantic_analyzer is not None
            variations = generate_variations(
                template,
                category,
                num_variations,
                use_ollama=use_semantic_for_this,
                semantic_analyzer=semantic_analyzer,
            )

            for variation in variations:
                if len(category_data) >= num_examples:
                    break

                fixed_text = fix_bare_verb_at_end(variation)
                if not is_valid_sentence(fixed_text):
                    continue

                task_id = f"task_{len(all_data) + len(category_data):06d}"
                example = {
                    "id": task_id,
                    "text": fixed_text,
                    "label": category,
                    "label_id": label_id,
                    "metadata": {
                        "length": len(fixed_text),
                        "difficulty": random.choice(["beginner", "intermediate", "advanced"]),
                        "source": "synthetic",
                    },
                }

                category_data.append(example)
                label_counts[category] += 1

        all_data.extend(category_data)

    random.shuffle(all_data)

    total = len(all_data)
    train_size = int(total * 0.70)
    val_size = int(total * 0.15)

    train_data = all_data[:train_size]
    val_data = all_data[train_size : train_size + val_size]
    test_data = all_data[train_size + val_size :]

    train_file = output_path / "classification_train.jsonl"
    val_file = output_path / "classification_val.jsonl"
    test_file = output_path / "classification_test.jsonl"

    for file_path, data in [(train_file, train_data), (val_file, val_data), (test_file, test_data)]:
        with open(file_path, "w") as f:
            for item in data:
                f.write(json.dumps({"text": item["text"], "label": item["label_id"]}) + "\n")

    full_train_file = output_path / "synthetic_classification_train.jsonl"
    full_val_file = output_path / "synthetic_classification_val.jsonl"
    full_test_file = output_path / "synthetic_classification_test.jsonl"

    for file_path, data in [
        (full_train_file, train_data),
        (full_val_file, val_data),
        (full_test_file, test_data),
    ]:
        with open(file_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    label_map_file = output_path / "label_mapping.json"
    with open(label_map_file, "w") as f:
        json.dump({"label2id": LABEL_MAPPING, "id2label": ID_TO_LABEL}, f, indent=2)

    metadata = {
        "dataset_name": "deepiri-task-classification-v1",
        "version": "1.0",
        "total_samples": total,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "num_classes": len(LABEL_MAPPING),
        "label_distribution": dict(label_counts),
        "avg_text_length": sum(len(item["text"]) for item in all_data) / len(all_data) if all_data else 0,
        "min_text_length": min((len(item["text"]) for item in all_data), default=0),
        "max_text_length": max((len(item["text"]) for item in all_data), default=0),
    }

    return {"train": train_data, "val": val_data, "test": test_data, "metadata": metadata}

