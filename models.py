# Copyright (c) 2026 - Personalized Learning Path OpenEnv Environment
"""
Pydantic models for the Personalized Learning Path environment.

Action  → choose a topic, a difficulty level, and a learning strategy.
Observation → full student state: per-topic mastery, cognitive load, fatigue,
              available/locked topics, session info, and task goal.
"""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


# ─────────────────────────────────────────────────────────────────────────────
# ACTION
# ─────────────────────────────────────────────────────────────────────────────

class LearningAction(Action):
    """
    One learning session decision.

    The agent picks a topic to study together with a difficulty level and a
    pedagogical strategy.

    Guidelines
    ----------
    - Only choose topics from ``available_topics`` (prerequisites met).
    - Match difficulty to current mastery:
        easy    → mastery < 0.30
        medium  → 0.30 ≤ mastery < 0.70
        hard    → mastery ≥ 0.70
    - Match strategy to learning stage:
        new_concept  → first exposure   (mastery < 0.20)
        practice     → building skill   (mastery 0.20–0.60)
        revision     → consolidating    (mastery 0.50–0.80)
        assessment   → testing mastery  (mastery > 0.60)
    """

    topic: str = Field(
        ...,
        description="Name of the topic to study. Must be in available_topics.",
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        ...,
        description=(
            "Difficulty level matched to current mastery: "
            "easy (<0.30), medium (0.30-0.70), hard (>0.70)."
        ),
    )
    strategy: Literal["new_concept", "practice", "revision", "assessment"] = Field(
        ...,
        description=(
            "Learning strategy: new_concept for first exposure, practice for "
            "skill building, revision for consolidation, assessment for testing."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# OBSERVATION
# ─────────────────────────────────────────────────────────────────────────────

class LearningObservation(Observation):
    """
    Full observation returned after every reset() and step().

    The observation carries the complete student state so the agent can make
    an informed decision about the next learning action.
    """

    # Task context
    task_id: str = Field(default="", description="Identifier of the active task.")
    goal: str = Field(default="", description="Plain-English description of the episode goal.")

    # Session progress
    session_number: int = Field(default=0, description="Sessions completed so far.", ge=0)
    time_remaining: int = Field(default=0, description="Sessions remaining before episode ends.", ge=0)

    # Student state
    topic_states: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-topic: {mastery: float, sessions_spent: int, last_strategy: str|null}.",
    )
    cognitive_load: float = Field(
        default=0.2, description="Cognitive load 0.0-1.0. High load reduces learning.", ge=0.0, le=1.0,
    )
    fatigue: float = Field(
        default=0.0, description="Accumulated fatigue 0.0-1.0. High fatigue reduces mastery gains.", ge=0.0, le=1.0,
    )

    # Action space hints
    available_topics: List[str] = Field(
        default_factory=list, description="Topics whose prerequisites are met (mastery >= 0.40).",
    )
    locked_topics: List[str] = Field(
        default_factory=list, description="Topics requiring further prerequisite progress.",
    )

    # Step feedback
    last_action_result: str = Field(default="", description="Human-readable outcome of the last action.")

    # Episode summary (populated when done=True)
    final_score: Optional[float] = Field(default=None, description="Episode score 0.0-1.0, set when done=True.")
    topics_mastered: int = Field(default=0, description="Target topics that reached target mastery.", ge=0)
