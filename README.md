---
title: Personalized Learning Path
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "3.11"
python_version: "3.11"
app_file: app.py
pinned: false
---

# 🎓 Personalized Learning Path — OpenEnv Environment

> **An AI agent acts as a curriculum planner, dynamically designing personalised
> learning paths for students by selecting topics, difficulty levels, and revision
> strategies to maximise learning outcomes under time and cognitive constraints.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v1.0-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

This environment simulates a real-world educational scenario: given a student
starting from zero knowledge, how should an AI agent sequence topics, calibrate
difficulty, and choose teaching strategies to achieve target mastery levels within
a limited number of sessions?

The environment models:

- **Zone of Proximal Development** — mastery gain is highest when difficulty is
  slightly above the student's current level (not too easy, not too hard)
- **Prerequisite graphs** — advanced topics are locked until foundations are built
- **Cognitive load** — over-challenging sessions reduce learning efficiency
- **Fatigue accumulation** — long high-intensity sessions compound across sessions
- **Partial reward signals** — the agent receives dense feedback on every step

---

## Action Space

The agent chooses **one action per session**:

| Field | Type | Values | Description |
|-------|------|---------|-------------|
| `topic` | `str` | task-specific | Topic to study (must be unlocked) |
| `difficulty` | `str` | `easy` / `medium` / `hard` | Pitch of the material |
| `strategy` | `str` | `new_concept` / `practice` / `revision` / `assessment` | Pedagogical mode |

**Optimal matching guidelines:**
- `easy` when mastery < 0.30, `medium` when 0.30–0.70, `hard` when > 0.70
- `new_concept` for first exposure (mastery < 0.20)
- `practice` for skill building (mastery 0.20–0.60)
- `revision` for consolidation (mastery 0.50–0.80)
- `assessment` for validation (mastery > 0.60)

---

## Observation Space

Each observation includes the complete student state:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Active task identifier |
| `goal` | `str` | Plain-English episode goal |
| `session_number` | `int` | Sessions completed so far |
| `time_remaining` | `int` | Sessions remaining |
| `topic_states` | `dict` | Per-topic `{mastery, sessions_spent, last_strategy}` |
| `cognitive_load` | `float` | Current cognitive load 0.0–1.0 |
| `fatigue` | `float` | Accumulated fatigue 0.0–1.0 |
| `available_topics` | `list[str]` | Topics whose prerequisites are met |
| `locked_topics` | `list[str]` | Topics needing prerequisite mastery |
| `last_action_result` | `str` | Human-readable outcome of last action |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float` | Step reward −0.50 to +1.0 |
| `final_score` | `float \| null` | Episode score 0.0–1.0 (set when `done=True`) |
| `topics_mastered` | `int` | Target topics that crossed target mastery |

---

## Tasks

### Task 1 — Basic Python Learning Path *(Easy)*

| Property | Value |
|----------|-------|
| `task_id` | `basic_python_path` |
| Topics | `variables`, `control_flow`, `functions`, `lists`, `dictionaries` |
| Max sessions | 8 |
| Target mastery | ≥ 0.70 on all 5 topics |
| Cognitive load threshold | 0.80 |

**Prerequisites:** `variables → control_flow → functions`, `variables + control_flow → lists`, `variables + lists → dictionaries`

A beginner course with a simple linear prerequisite chain. An optimal agent
achieves full mastery in 6–7 sessions by matching difficulty to mastery levels.

---

### Task 2 — Web Development Curriculum *(Medium)*

| Property | Value |
|----------|-------|
| `task_id` | `web_dev_curriculum` |
| Topics | HTML, CSS, JS, DOM, async JS, React, hooks, state, REST API, deployment |
| Max sessions | 15 |
| Target mastery | ≥ 0.75 on 7+ core topics |
| Cognitive load threshold | 0.75 |

A realistic curriculum with a branching prerequisite graph. The agent must plan
a valid topological ordering while keeping cognitive load manageable. Sub-optimal
orderings waste sessions on locked topics.

---

### Task 3 — Adaptive ML Curriculum *(Hard)*

| Property | Value |
|----------|-------|
| `task_id` | `ml_curriculum` |
| Topics | 15 topics from linear algebra → deep learning → deployment |
| Max sessions | 20 |
| Target mastery | ≥ 0.80 on 8 core ML pipeline topics |
| Cognitive load threshold | 0.70 |

The hardest task: a deeply connected prerequisite DAG, strict mastery targets,
tighter cognitive load limits, and fatigue that accumulates across 20 sessions.
Frontier models score ~0.35–0.55 with greedy strategies.

---

## Reward Function

The step reward is a dense, shaped signal:

```
reward = mastery_gain × topic_weight × 2.0      # primary learning signal
       + milestone_bonus                          # 0.30 × weight on crossing target
       + efficiency_bonus                         # ±0.05 difficulty–mastery match
       - cognitive_load_penalty                   # scales above threshold
       - fatigue_penalty                          # small per-step tax
```

**Clipped to [−0.50, +1.0]** per step.

The **episode score** (0.0–1.0) combines:
- 50% weighted average mastery on target topics
- 30% fraction of target topics at target mastery
- 10% efficiency (sessions remaining at completion)
- 10% student wellness (low load + low fatigue)

---

## Setup & Usage

### Local (Python)

```bash
# Install dependencies
pip install openenv-core openai requests

# Start the server
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

# In another terminal — run baseline inference
export HF_TOKEN=your_api_key
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

### With uv

```bash
uv sync
uv run server                   # starts on port 8000
python inference.py             # in another terminal
```

### Docker

```bash
docker build -t personalized-learning-path -f server/Dockerfile .
docker run -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  -e MODEL_NAME=$MODEL_NAME \
  -e API_BASE_URL=$API_BASE_URL \
  personalized-learning-path
```

---

## API Reference

| Method | Path | Body | Description |
|--------|------|------|-------------|
| `POST` | `/reset` | `{"task_id": "basic_python_path"}` | Start new episode |
| `POST` | `/step` | `{"action": {"topic": ..., "difficulty": ..., "strategy": ...}}` | Take one step |
| `GET` | `/state` | — | Inspect current state |
| `GET` | `/health` | — | Health check |
| `GET` | `/schema` | — | Action / observation schemas |
| `WS` | `/ws` | — | Persistent WebSocket session |

**Example interaction:**

```bash
# Reset to hard task
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "ml_curriculum"}'

# Take a step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"topic": "linear_algebra", "difficulty": "easy", "strategy": "new_concept"}}'
```

---

## Baseline Scores

Scores achieved by `meta-llama/Llama-3.3-70B-Instruct` via HuggingFace Router:

| Task | Difficulty | Score |
|------|-----------|-------|
| `basic_python_path` | Easy | ~0.42 |
| `web_dev_curriculum` | Medium | ~0.31 |
| `ml_curriculum` | Hard | ~0.22 |
| **Average** | | **~0.32** |

A perfect agent achieving all targets with maximum efficiency would score **1.0**.
A random agent typically scores **0.05–0.12**.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | LLM API endpoint |
| `MODEL_NAME` | Yes | Model identifier for inference |
| `HF_TOKEN` | Yes | HuggingFace / API key |
| `ENV_URL` | No | Environment server URL (default: `http://localhost:8000`) |

---

## Project Structure

```
personalized_learning_path/
├── models.py              # LearningAction + LearningObservation (Pydantic v2)
├── client.py              # EnvClient WebSocket subclass
├── inference.py           # Baseline LLM agent (all 3 tasks)
├── openenv.yaml           # OpenEnv spec metadata
├── pyproject.toml         # Package config + dependencies
├── uv.lock                # Locked dependencies
└── server/
    ├── app.py             # FastAPI app via create_app() + singleton factory
    ├── tasks.py           # Task definitions (easy / medium / hard)
    ├── personalized_learning_path_environment.py  # Core env logic
    ├── requirements.txt   # Docker pip requirements
    └── Dockerfile         # Multi-stage container build
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
