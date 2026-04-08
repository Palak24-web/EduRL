# Copyright (c) 2026 - Personalized Learning Path OpenEnv Environment
"""
Personalized Learning Path Environment — core implementation.

Reward shaping
--------------
  +  mastery_gain x topic_weight x 2.0       (primary learning signal)
  +  milestone_bonus (0.30 x weight)          (crossing target_mastery)
  +  efficiency_bonus (+-0.05)                (good difficulty/mastery match)
  -  cognitive_load_penalty                   (load > threshold)
  -  fatigue_penalty                          (accumulated fatigue)
  -  0.05 penalty for locked/invalid topic
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import LearningAction, LearningObservation
    from .tasks import DEFAULT_TASK_ID, PREREQ_THRESHOLD, TASKS, TaskDefinition
except (ImportError, ModuleNotFoundError):
    from models import LearningAction, LearningObservation
    from server.tasks import DEFAULT_TASK_ID, PREREQ_THRESHOLD, TASKS, TaskDefinition


class PersonalizedLearningPathEnvironment(
    Environment[LearningAction, LearningObservation, State]
):
    """OpenEnv environment for AI-driven personalised learning path optimisation."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._task: Optional[TaskDefinition] = None
        self._topic_mastery: Dict[str, float] = {}
        self._topic_sessions: Dict[str, int] = {}
        self._topic_strategy: Dict[str, Optional[str]] = {}
        self._cognitive_load: float = 0.2
        self._fatigue: float = 0.0
        self._sessions_done: int = 0
        self._done: bool = False
        self._episode_id: str = str(uuid4())
        self._episode_rewards: List[float] = []

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> LearningObservation:
        """Reset the environment for a new episode."""
        self._reset_rubric()
        tid = task_id or DEFAULT_TASK_ID
        if tid not in TASKS:
            tid = DEFAULT_TASK_ID

        self._task = TASKS[tid]
        self._topic_mastery = {t: 0.0 for t in self._task.topics}
        self._topic_sessions = {t: 0 for t in self._task.topics}
        self._topic_strategy = {t: None for t in self._task.topics}
        self._cognitive_load = 0.20
        self._fatigue = 0.00
        self._sessions_done = 0
        self._done = False
        self._episode_id = episode_id or str(uuid4())
        self._episode_rewards = []

        available, locked = self._split_topics()
        return LearningObservation(
            task_id=self._task.id,
            goal=self._task.goal_description,
            session_number=0,
            time_remaining=self._task.max_sessions,
            topic_states=self._build_topic_states(),
            cognitive_load=self._cognitive_load,
            fatigue=self._fatigue,
            available_topics=available,
            locked_topics=locked,
            last_action_result="Environment reset - ready to start learning!",
            done=False,
            reward=0.0,
            topics_mastered=0,
        )

    def step(
        self,
        action: LearningAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> LearningObservation:
        """Execute one learning session."""
        if self._task is None:
            self.reset()

        if self._done:
            return self._build_observation(
                reward=0.0,
                result_msg="Episode is done. Call /reset to start a new episode.",
            )

        available, locked = self._split_topics()

        # Validate action
        if action.topic not in self._task.topics:
            reward = -0.10
            result_msg = (
                f"Invalid topic '{action.topic}'. "
                f"Valid topics: {self._task.topics}."
            )
        elif action.topic in locked:
            reward = -0.05
            prereqs = self._task.prerequisites.get(action.topic, [])
            result_msg = (
                f"Topic '{action.topic}' is LOCKED. "
                f"Prerequisite topics (mastery >= {PREREQ_THRESHOLD}): {prereqs}."
            )
        else:
            reward, result_msg = self._apply_learning(action)

        self._sessions_done += 1
        self._episode_rewards.append(reward)
        self._done = self._check_done()

        obs = self._build_observation(reward=reward, result_msg=result_msg)
        return self._apply_transform(obs)

    @property
    def state(self) -> State:
        return State(episode_id=self._episode_id, step_count=self._sessions_done)

    def close(self) -> None:
        """No-op: singleton keeps state across HTTP calls."""
        pass

    # -------------------------------------------------------------------------
    # Learning dynamics
    # -------------------------------------------------------------------------

    def _apply_learning(self, action: LearningAction) -> Tuple[float, str]:
        topic = action.topic
        cur_mast = self._topic_mastery[topic]
        difficulty = action.difficulty
        strategy = action.strategy

        gain = self._mastery_gain(
            cur_mast, difficulty, strategy, self._cognitive_load, self._fatigue
        )
        new_mast = min(1.0, cur_mast + gain)
        self._topic_mastery[topic] = new_mast
        self._topic_sessions[topic] += 1
        self._topic_strategy[topic] = strategy

        self._cognitive_load = self._update_load(
            self._cognitive_load, difficulty, strategy, cur_mast
        )
        self._fatigue = self._update_fatigue(self._fatigue, self._cognitive_load)

        w = self._task.weights.get(topic, 1.0)
        mastery_reward = gain * w * 2.0
        milestone_bonus = 0.30 * w if (cur_mast < self._task.target_mastery <= new_mast) else 0.0
        efficiency_bonus = self._efficiency_bonus(cur_mast, difficulty)
        load_penalty = max(
            0.0, (self._cognitive_load - self._task.cognitive_load_penalty_threshold) * 0.50
        )
        fatigue_penalty = self._fatigue * 0.05
        reward = float(
            max(-0.50, min(1.0,
                mastery_reward + milestone_bonus + efficiency_bonus
                - load_penalty - fatigue_penalty
            ))
        )

        msg = (
            f"Studied '{topic}' | {difficulty} | {strategy}. "
            f"Mastery: {cur_mast:.2f} -> {new_mast:.2f} (+{gain:.3f}). "
            f"Load: {self._cognitive_load:.2f} | Fatigue: {self._fatigue:.2f}."
        )
        if milestone_bonus > 0:
            msg += f" [MILESTONE] '{topic}' reached target mastery!"
        if self._cognitive_load > self._task.cognitive_load_penalty_threshold:
            msg += " [WARNING] High cognitive load - consider easier difficulty or revision."

        return reward, msg

    @staticmethod
    def _mastery_gain(
        mastery: float,
        difficulty: str,
        strategy: str,
        load: float,
        fatigue: float,
    ) -> float:
        """Zone of Proximal Development mastery gain model."""
        if difficulty == "easy":
            if mastery < 0.30:
                base = 0.22
            elif mastery < 0.60:
                base = 0.12
            else:
                base = 0.05
        elif difficulty == "medium":
            if mastery < 0.25:
                base = 0.07
            elif mastery < 0.55:
                base = 0.20
            elif mastery < 0.80:
                base = 0.17
            else:
                base = 0.07
        else:  # hard
            if mastery < 0.45:
                base = 0.03
            elif mastery < 0.70:
                base = 0.15
            elif mastery < 0.90:
                base = 0.20
            else:
                base = 0.07

        strategy_mult = {
            "new_concept": 1.30 if mastery < 0.40 else 0.65,
            "practice":    1.00,
            "revision":    0.55 if mastery < 0.40 else 1.20,
            "assessment":  0.35 if mastery < 0.55 else 1.30,
        }.get(strategy, 1.0)

        gain = base * strategy_mult

        if load > 0.80:
            gain *= max(0.10, 1.0 - (load - 0.80) * 3.0)
        elif load > 0.60:
            gain *= 1.0 - (load - 0.60) * 0.50

        gain *= max(0.20, 1.0 - fatigue * 0.60)

        return float(max(0.0, min(gain, 1.0 - mastery)))

    @staticmethod
    def _update_load(
        load: float, difficulty: str, strategy: str, mastery: float
    ) -> float:
        diff_delta = {"easy": -0.08, "medium": 0.05, "hard": 0.15}[difficulty]
        strat_delta = {
            "new_concept": 0.12,
            "practice": 0.02,
            "revision": -0.08,
            "assessment": 0.05,
        }[strategy]
        mismatch = 0.0
        if difficulty == "hard" and mastery < 0.40:
            mismatch = 0.18
        elif difficulty == "medium" and mastery < 0.20:
            mismatch = 0.10
        recovery = (load - 0.20) * 0.05
        return float(max(0.05, min(1.0, load + diff_delta + strat_delta + mismatch - recovery)))

    @staticmethod
    def _update_fatigue(fatigue: float, load: float) -> float:
        return float(max(0.0, min(1.0, fatigue + 0.04 + load * 0.06 - 0.01)))

    @staticmethod
    def _efficiency_bonus(mastery: float, difficulty: str) -> float:
        optimal = {"easy": 0.15, "medium": 0.45, "hard": 0.72}[difficulty]
        dist = abs(mastery - optimal)
        if dist < 0.15:
            return 0.05
        elif dist < 0.30:
            return 0.02
        else:
            return -0.02

    # -------------------------------------------------------------------------
    # Topic availability
    # -------------------------------------------------------------------------

    def _split_topics(self) -> Tuple[List[str], List[str]]:
        if self._task is None:
            return [], []
        available, locked = [], []
        for topic in self._task.topics:
            prereqs = self._task.prerequisites.get(topic, [])
            if all(self._topic_mastery.get(p, 0.0) >= PREREQ_THRESHOLD for p in prereqs):
                available.append(topic)
            else:
                locked.append(topic)
        return available, locked

    # -------------------------------------------------------------------------
    # Termination & scoring
    # -------------------------------------------------------------------------

    def _check_done(self) -> bool:
        if self._task is None:
            return True
        if self._sessions_done >= self._task.max_sessions:
            return True
        mastered = sum(
            1 for t in self._task.target_topics
            if self._topic_mastery.get(t, 0.0) >= self._task.target_mastery
        )
        return mastered >= self._task.min_topics_to_complete

    def _compute_final_score(self) -> float:
        if self._task is None:
            return 0.0
        target = self._task.target_topics
        total_w = sum(self._task.weights.get(t, 1.0) for t in target) or 1.0
        weighted_mastery = sum(
            self._topic_mastery.get(t, 0.0) * self._task.weights.get(t, 1.0)
            for t in target
        ) / total_w
        mastered_count = sum(
            1 for t in target
            if self._topic_mastery.get(t, 0.0) >= self._task.target_mastery
        )
        completion = mastered_count / max(1, len(target))
        efficiency = max(0.0, self._task.max_sessions - self._sessions_done) / self._task.max_sessions
        wellness = (1.0 - self._cognitive_load) * 0.5 + (1.0 - self._fatigue) * 0.5
        score = (
            weighted_mastery * 0.50
            + completion     * 0.30
            + efficiency     * 0.10
            + wellness       * 0.10
        )
        return float(max(0.0, min(1.0, score)))

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _build_topic_states(self) -> Dict[str, Dict]:
        return {
            topic: {
                "mastery": round(self._topic_mastery.get(topic, 0.0), 4),
                "sessions_spent": self._topic_sessions.get(topic, 0),
                "last_strategy": self._topic_strategy.get(topic),
            }
            for topic in (self._task.topics if self._task else [])
        }

    def _count_mastered(self) -> int:
        if self._task is None:
            return 0
        return sum(
            1 for t in self._task.target_topics
            if self._topic_mastery.get(t, 0.0) >= self._task.target_mastery
        )

    def _build_observation(self, reward: float, result_msg: str) -> LearningObservation:
        available, locked = self._split_topics()
        mastered = self._count_mastered()
        final_score: Optional[float] = self._compute_final_score() if self._done else None
        return LearningObservation(
            task_id=self._task.id if self._task else "",
            goal=self._task.goal_description if self._task else "",
            session_number=self._sessions_done,
            time_remaining=max(
                0, (self._task.max_sessions if self._task else 0) - self._sessions_done
            ),
            topic_states=self._build_topic_states(),
            cognitive_load=round(self._cognitive_load, 4),
            fatigue=round(self._fatigue, 4),
            available_topics=available,
            locked_topics=locked,
            last_action_result=result_msg,
            done=self._done,
            reward=reward,
            topics_mastered=mastered,
            final_score=final_score,
        )
