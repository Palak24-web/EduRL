#!/usr/bin/env python3
# Copyright (c) 2026 - Personalized Learning Path OpenEnv Environment
"""
Baseline Inference Script — Personalized Learning Path Environment
===================================================================
Runs an LLM-based agent against all 3 tasks and reports scores.

Environment variables required:
    API_BASE_URL  — LLM API endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME    — Model identifier  (default: meta-llama/Llama-3.3-70B-Instruct)
    HF_TOKEN      — HuggingFace / API key
    ENV_URL       — Environment server URL (default: http://localhost:8000)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:8000")

MAX_STEPS    = 25   # safety cap per episode
TEMPERATURE  = 0.10
MAX_TOKENS   = 200

# ── OpenAI client ─────────────────────────────────────────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert curriculum designer. Your goal is to maximise a student's
learning outcomes by choosing the best topic, difficulty, and strategy each session.

You will receive the student's current state and must respond with ONLY a JSON object:
{
  "topic": "<topic name from available_topics>",
  "difficulty": "<easy|medium|hard>",
  "strategy": "<new_concept|practice|revision|assessment>"
}

Decision rules:
- ONLY pick topics from available_topics (not locked_topics).
- Match difficulty to current mastery:
    easy    if mastery < 0.30
    medium  if 0.30 <= mastery < 0.70
    hard    if mastery >= 0.70
- Match strategy to learning stage:
    new_concept  if mastery < 0.20  (first exposure)
    practice     if 0.20-0.60       (skill building)
    revision     if 0.50-0.80       (consolidation)
    assessment   if mastery > 0.65  (testing)
- If cognitive_load > 0.70 prefer easier difficulty or revision strategy.
- Prioritise topics that are prerequisites for locked topics.
- Respond ONLY with the JSON object — no extra text, no markdown."""


def build_user_prompt(obs: Dict[str, Any]) -> str:
    """Convert an observation dict into a clear prompt for the LLM."""
    topic_masteries = {
        t: round(v.get("mastery", 0.0), 2)
        for t, v in obs.get("topic_states", {}).items()
    }
    return (
        f"Task: {obs.get('goal', '')}\n"
        f"Session: {obs.get('session_number', 0)} | "
        f"Sessions remaining: {obs.get('time_remaining', 0)}\n"
        f"Cognitive load: {obs.get('cognitive_load', 0):.2f} | "
        f"Fatigue: {obs.get('fatigue', 0):.2f}\n\n"
        f"available_topics: {obs.get('available_topics', [])}\n"
        f"locked_topics:    {obs.get('locked_topics', [])}\n\n"
        f"Topic mastery:\n{json.dumps(topic_masteries, indent=2)}\n\n"
        f"Last result: {obs.get('last_action_result', '')}\n\n"
        "Choose the next learning action as a JSON object."
    )


def pick_fallback_action(obs: Dict[str, Any]) -> Dict[str, str]:
    """Heuristic fallback when the LLM fails to produce a valid action."""
    available = obs.get("available_topics", [])
    if not available:
        available = list(obs.get("topic_states", {}).keys())[:1]
    if not available:
        return {"topic": "variables", "difficulty": "easy", "strategy": "new_concept"}

    topic_states = obs.get("topic_states", {})
    # Pick the available topic with the lowest mastery (most to gain)
    topic = min(available, key=lambda t: topic_states.get(t, {}).get("mastery", 0.0))
    mastery = topic_states.get(topic, {}).get("mastery", 0.0)

    load = obs.get("cognitive_load", 0.2)
    if load > 0.75:
        difficulty = "easy"
        strategy   = "revision" if mastery > 0.4 else "practice"
    elif mastery < 0.25:
        difficulty = "easy"
        strategy   = "new_concept"
    elif mastery < 0.55:
        difficulty = "medium"
        strategy   = "practice"
    elif mastery < 0.75:
        difficulty = "medium"
        strategy   = "revision"
    else:
        difficulty = "hard"
        strategy   = "assessment"

    return {"topic": topic, "difficulty": difficulty, "strategy": strategy}


def get_llm_action(obs: Dict[str, Any]) -> Dict[str, str]:
    """Ask the LLM for the next action; fall back to heuristic on any error."""
    available = obs.get("available_topics", [])
    try:
        completion = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(obs)},
            ],
            temperature = TEMPERATURE,
            max_tokens  = MAX_TOKENS,
        )
        raw = completion.choices[0].message.content or ""
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        action = json.loads(raw.strip())

        # Validate fields
        assert action.get("topic")      in available,                      "topic not in available"
        assert action.get("difficulty") in ("easy", "medium", "hard"),     "bad difficulty"
        assert action.get("strategy")   in (
            "new_concept", "practice", "revision", "assessment"
        ),                                                                  "bad strategy"
        return {
            "topic":      action["topic"],
            "difficulty": action["difficulty"],
            "strategy":   action["strategy"],
        }

    except Exception as exc:
        print(f"    [LLM fallback] {type(exc).__name__}: {exc}")
        return pick_fallback_action(obs)


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def http_reset(task_id: str) -> Dict[str, Any]:
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    resp = r.json()
    # create_app returns {"observation": {...}, "done": bool, "reward": ...}
    return resp.get("observation", resp)


def http_step(action: Dict[str, str]) -> Tuple[Dict[str, Any], float, bool]:
    r = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=30)
    r.raise_for_status()
    resp = r.json()
    obs  = resp.get("observation", resp)
    reward = float(resp.get("reward") or 0.0)
    done   = bool(resp.get("done", obs.get("done", False)))
    return obs, reward, done


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(task_id: str) -> float:
    """Run one episode and return the final score (0.0–1.0)."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Task: {task_id}")
    print(sep)

    obs = http_reset(task_id)
    print(f"Goal: {obs.get('goal', '(unknown)')}")

    # Structured output: episode start (required by validator)
    print(f"[START] task={task_id}", flush=True)

    total_reward = 0.0
    final_score  = 0.0
    step = 0

    for step in range(1, MAX_STEPS + 1):
        if obs.get("done", False):
            print(f"  Done signalled at step {step - 1}.")
            break

        action = get_llm_action(obs)
        print(
            f"  Step {step:2d}: {action['topic']:<20} | "
            f"{action['difficulty']:<6} | {action['strategy']}"
        )

        obs, reward, done = http_step(action)
        total_reward += reward
        mastered = obs.get("topics_mastered", 0)
        print(
            f"           reward={reward:+.3f} | total={total_reward:+.3f} | "
            f"mastered={mastered} | load={obs.get('cognitive_load', 0):.2f}"
        )

        # Structured output: per-step block (required by validator)
        print(f"[STEP] step={step} reward={round(reward, 4)}", flush=True)

        if done:
            final_score = obs.get("final_score") or 0.0
            print(f"  Episode complete! Final score: {final_score:.4f}")
            break
    else:
        # Max steps reached — compute score from last obs
        final_score = obs.get("final_score") or 0.0
        print(f"  Reached max steps ({MAX_STEPS}). Score: {final_score:.4f}")

    # Structured output: episode end (required by validator)
    print(f"[END] task={task_id} score={round(final_score, 4)} steps={step}", flush=True)

    return float(final_score)


# ── Server management ─────────────────────────────────────────────────────────

def wait_for_server(url: str, timeout: int = 60) -> bool:
    """Poll the server until it responds or timeout."""
    print(f"Waiting for server at {url} ...", end="", flush=True)
    for _ in range(timeout):
        try:
            r = requests.get(url, timeout=3)
            if r.status_code < 500:
                print(" ready.")
                return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(1)
    print(" TIMEOUT")
    return False


def maybe_start_server() -> Optional[subprocess.Popen]:
    """Start a local server subprocess if ENV_URL is localhost and not running."""
    try:
        requests.get(ENV_URL, timeout=3)
        print("Server already running.")
        return None
    except Exception:
        pass

    if "localhost" not in ENV_URL and "127.0.0.1" not in ENV_URL:
        print(f"Remote server {ENV_URL} is not responding.")
        return None

    print("Starting local server ...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app",
         "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not wait_for_server(ENV_URL, timeout=60):
        proc.terminate()
        print("Failed to start server — exiting.")
        sys.exit(1)
    return proc


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Personalized Learning Path — Baseline Inference        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"Model : {MODEL_NAME}")
    print(f"API   : {API_BASE_URL}")
    print(f"Server: {ENV_URL}")

    server_proc = maybe_start_server()

    task_ids: List[str] = [
        "basic_python_path",
        "web_dev_curriculum",
        "ml_curriculum",
    ]
    scores: Dict[str, float] = {}

    try:
        for tid in task_ids:
            try:
                scores[tid] = run_episode(tid)
            except Exception as exc:
                print(f"  [ERROR] Task '{tid}' failed: {exc}")
                scores[tid] = 0.0

        avg = sum(scores.values()) / len(scores) if scores else 0.0

        print("\n" + "=" * 60)
        print("BASELINE SCORES")
        print("=" * 60)
        for tid, score in scores.items():
            bar = "█" * int(score * 20)
            print(f"  {tid:<30} {score:.4f}  |{bar:<20}|")
        print(f"  {'Average':<30} {avg:.4f}")
        print("=" * 60)

    finally:
        if server_proc is not None:
            server_proc.terminate()
            server_proc.wait(timeout=5)


if __name__ == "__main__":
    main()
