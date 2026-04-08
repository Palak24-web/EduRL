# Copyright (c) 2026 - Personalized Learning Path OpenEnv Environment
"""Personalized Learning Path Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import LearningAction, LearningObservation
except ImportError:
    from models import LearningAction, LearningObservation


class PersonalizedLearningPathEnv(
    EnvClient[LearningAction, LearningObservation, State]
):
    """
    Async WebSocket client for the Personalized Learning Path Environment.

    Usage (async):
        async with PersonalizedLearningPathEnv(base_url="http://localhost:8000") as env:
            result = await env.reset(task_id="basic_python_path")
            obs = result.observation
            result = await env.step(
                LearningAction(topic="variables", difficulty="easy", strategy="new_concept")
            )

    Usage (sync):
        with PersonalizedLearningPathEnv(base_url="http://localhost:8000").sync() as env:
            result = env.reset(task_id="basic_python_path")
            result = env.step(
                LearningAction(topic="variables", difficulty="easy", strategy="new_concept")
            )
    """

    def _step_payload(self, action: LearningAction) -> Dict[str, Any]:
        return {
            "topic":      action.topic,
            "difficulty": action.difficulty,
            "strategy":   action.strategy,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[LearningObservation]:
        obs_data = payload.get("observation", {})
        observation = LearningObservation(
            task_id            = obs_data.get("task_id", ""),
            goal               = obs_data.get("goal", ""),
            session_number     = obs_data.get("session_number", 0),
            time_remaining     = obs_data.get("time_remaining", 0),
            topic_states       = obs_data.get("topic_states", {}),
            cognitive_load     = obs_data.get("cognitive_load", 0.2),
            fatigue            = obs_data.get("fatigue", 0.0),
            available_topics   = obs_data.get("available_topics", []),
            locked_topics      = obs_data.get("locked_topics", []),
            last_action_result = obs_data.get("last_action_result", ""),
            done               = payload.get("done", False),
            reward             = payload.get("reward"),
            topics_mastered    = obs_data.get("topics_mastered", 0),
            final_score        = obs_data.get("final_score"),
            metadata           = obs_data.get("metadata", {}),
        )
        return StepResult(
            observation = observation,
            reward      = payload.get("reward"),
            done        = payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id = payload.get("episode_id"),
            step_count = payload.get("step_count", 0),
        )
