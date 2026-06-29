"""Intent classifier label vocabulary (31 categories)."""

from __future__ import annotations

from typing import Dict

CATEGORIES: Dict[int, str] = {
    0: "debugging",
    1: "refactoring",
    2: "writing_code",
    3: "programming",
    4: "running_code",
    5: "inspecting",
    6: "writing",
    7: "learning_research",
    8: "learning_study",
    9: "learning_training",
    10: "learning_practice",
    11: "creative",
    12: "administrative",
    13: "team_organization",
    14: "team_collaboration",
    15: "team_planning",
    16: "research",
    17: "planning",
    18: "communication",
    19: "big_data_analytics",
    20: "data_processing",
    21: "design",
    22: "qa",
    23: "testing",
    24: "validation",
    25: "reporting",
    26: "documentation",
    27: "system_admin",
    28: "ux_ui",
    29: "security",
    30: "data_privacy",
}

LABEL_TO_ID: Dict[str, int] = {name: idx for idx, name in CATEGORIES.items()}
