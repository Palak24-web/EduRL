# Copyright (c) 2026 - Personalized Learning Path OpenEnv Environment
"""
Task definitions for the Personalized Learning Path environment.

Three tasks with increasing difficulty:
  - basic_python_path   (easy):   5 topics,  8 sessions
  - web_dev_curriculum  (medium): 10 topics, 15 sessions
  - ml_curriculum       (hard):   15 topics, 20 sessions
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TaskDefinition:
    id: str
    name: str
    difficulty: str  # easy | medium | hard
    description: str
    goal_description: str
    topics: List[str]
    # topic -> list of topics whose mastery must be >= PREREQ_THRESHOLD
    prerequisites: Dict[str, List[str]]
    max_sessions: int
    target_mastery: float
    # Topics that must reach target_mastery to win
    target_topics: List[str]
    min_topics_to_complete: int           # how many target_topics must reach target_mastery
    cognitive_load_penalty_threshold: float
    # Per-topic weight for scoring (more important topics score higher)
    weights: Dict[str, float]


# Mastery threshold required for a prerequisite to count as "met"
PREREQ_THRESHOLD: float = 0.40

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — EASY
# ─────────────────────────────────────────────────────────────────────────────
_BASIC_PYTHON = TaskDefinition(
    id="basic_python_path",
    name="Basic Python Learning Path",
    difficulty="easy",
    description=(
        "Guide a complete beginner through the five core Python programming "
        "concepts: variables, control flow, functions, lists, and dictionaries. "
        "No prerequisites between topics except a simple linear sequence."
    ),
    goal_description=(
        "Achieve mastery >= 0.70 on ALL 5 Python topics within 8 sessions. "
        "Manage cognitive load to maximise learning efficiency."
    ),
    topics=["variables", "control_flow", "functions", "lists", "dictionaries"],
    prerequisites={
        "variables":    [],
        "control_flow": ["variables"],
        "functions":    ["variables", "control_flow"],
        "lists":        ["variables", "control_flow"],
        "dictionaries": ["variables", "lists"],
    },
    max_sessions=8,
    target_mastery=0.70,
    target_topics=["variables", "control_flow", "functions", "lists", "dictionaries"],
    min_topics_to_complete=5,
    cognitive_load_penalty_threshold=0.80,
    weights={
        "variables":    1.0,
        "control_flow": 1.0,
        "functions":    1.2,
        "lists":        1.0,
        "dictionaries": 1.0,
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — MEDIUM
# ─────────────────────────────────────────────────────────────────────────────
_WEB_DEV = TaskDefinition(
    id="web_dev_curriculum",
    name="Web Development Curriculum",
    difficulty="medium",
    description=(
        "Navigate a 10-topic web development curriculum with prerequisite "
        "constraints and cognitive load limits. Topics range from HTML basics "
        "to React state management and REST API integration."
    ),
    goal_description=(
        "Achieve mastery >= 0.75 on at least 7 of the 10 core web topics "
        "within 15 sessions. Keep average cognitive load below 0.75."
    ),
    topics=[
        "html_basics", "css_styling", "js_fundamentals", "dom_manipulation",
        "async_js", "react_basics", "react_hooks", "state_management",
        "rest_api", "deployment",
    ],
    prerequisites={
        "html_basics":      [],
        "css_styling":      ["html_basics"],
        "js_fundamentals":  ["html_basics"],
        "dom_manipulation": ["js_fundamentals", "html_basics"],
        "async_js":         ["js_fundamentals"],
        "react_basics":     ["js_fundamentals", "dom_manipulation"],
        "react_hooks":      ["react_basics"],
        "state_management": ["react_hooks"],
        "rest_api":         ["async_js", "js_fundamentals"],
        "deployment":       ["rest_api", "react_basics"],
    },
    max_sessions=15,
    target_mastery=0.75,
    target_topics=[
        "html_basics", "css_styling", "js_fundamentals", "dom_manipulation",
        "react_basics", "react_hooks", "rest_api",
    ],
    min_topics_to_complete=7,
    cognitive_load_penalty_threshold=0.75,
    weights={
        "html_basics":      0.8,
        "css_styling":      0.8,
        "js_fundamentals":  1.3,
        "dom_manipulation": 1.0,
        "async_js":         1.0,
        "react_basics":     1.4,
        "react_hooks":      1.1,
        "state_management": 1.0,
        "rest_api":         1.2,
        "deployment":       0.9,
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — HARD
# ─────────────────────────────────────────────────────────────────────────────
_ML_CURRICULUM = TaskDefinition(
    id="ml_curriculum",
    name="Adaptive ML Curriculum",
    difficulty="hard",
    description=(
        "Design an optimal ML learning path across 15 topics with a complex "
        "prerequisite graph, strict fatigue management, and adaptive difficulty "
        "requirements. Topics span linear algebra → deep learning → deployment."
    ),
    goal_description=(
        "Complete the 8-topic core ML pipeline (mastery >= 0.80 each) within "
        "20 sessions. Manage fatigue and cognitive overload or risk diminishing "
        "learning returns."
    ),
    topics=[
        "linear_algebra", "probability", "statistics", "python_advanced",
        "numpy_pandas", "data_viz", "ml_fundamentals", "regression",
        "classification", "clustering", "neural_networks", "deep_learning",
        "nlp_basics", "model_evaluation", "model_deployment",
    ],
    prerequisites={
        "linear_algebra":    [],
        "probability":       [],
        "statistics":        ["probability"],
        "python_advanced":   [],
        "numpy_pandas":      ["python_advanced"],
        "data_viz":          ["numpy_pandas"],
        "ml_fundamentals":   ["statistics", "linear_algebra", "numpy_pandas"],
        "regression":        ["ml_fundamentals"],
        "classification":    ["ml_fundamentals"],
        "clustering":        ["ml_fundamentals"],
        "neural_networks":   ["regression", "classification", "linear_algebra"],
        "deep_learning":     ["neural_networks"],
        "nlp_basics":        ["deep_learning", "python_advanced"],
        "model_evaluation":  ["regression", "classification"],
        "model_deployment":  ["model_evaluation", "python_advanced"],
    },
    max_sessions=20,
    target_mastery=0.80,
    target_topics=[
        "linear_algebra", "probability", "statistics",
        "numpy_pandas", "ml_fundamentals", "regression",
        "classification", "model_evaluation",
    ],
    min_topics_to_complete=8,
    cognitive_load_penalty_threshold=0.70,
    weights={
        "linear_algebra":   1.0,
        "probability":      1.0,
        "statistics":       1.1,
        "python_advanced":  0.9,
        "numpy_pandas":     1.0,
        "data_viz":         0.7,
        "ml_fundamentals":  1.5,
        "regression":       1.2,
        "classification":   1.2,
        "clustering":       0.9,
        "neural_networks":  1.3,
        "deep_learning":    1.2,
        "nlp_basics":       1.0,
        "model_evaluation": 1.3,
        "model_deployment": 1.0,
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────
TASKS: Dict[str, TaskDefinition] = {
    _BASIC_PYTHON.id:    _BASIC_PYTHON,
    _WEB_DEV.id:         _WEB_DEV,
    _ML_CURRICULUM.id:   _ML_CURRICULUM,
}

DEFAULT_TASK_ID = _BASIC_PYTHON.id
