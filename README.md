---
title: Resume Screening Env
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
app_file: inference.py
pinned: false
tags:
  - openenv
---

# Resume Screening Environment

A real-world reinforcement learning environment where an agent acts as a recruiter —
screening and ranking candidates based on job requirements.
Implements the OpenEnv interface.

---

## Motivation

Resume screening is a high-volume and time-intensive task in modern recruitment. 
Hiring decisions are not based on simple keyword matching — they require evaluating 
candidates against weighted skill requirements, relevant experience, and overall fit.

In practice, recruiters must:
- Compare candidates based on **must-have and nice-to-have skills**
- Consider **experience alongside technical abilities**
- Identify **transferable skills** that may not directly match job requirements
- Prioritize candidates by overall relevance, not just individual attributes

This environment models resume screening as a ranking problem, where an agent must 
order candidates from best to worst fit for a given role. Unlike classification tasks, 
ranking requires understanding relative differences between candidates and making 
consistent decisions across the entire pool.

By framing this as a structured environment, we enable:
- Objective evaluation using ranking-based rewards
- Benchmarking of model reasoning on real-world decision tasks
- Robust handling of imperfect outputs such as missing or extra candidates

This creates a practical testbed for evaluating how well AI systems can assist in 
realistic hiring workflows.

---

## Environment Overview

| Property | Value |
|---|---|
| Interface | OpenEnv |
| Tasks | 3 (easy, medium, hard) |
| Steps per episode | 3 |
| Reward range | 0.0 – 1.0 |
| Action type | Ranked list of candidate IDs |
| Observation type | Job description + candidate pool |

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| job_description | dict | Role, must-have skills (weighted), nice-to-have skills, min experience |
| candidates | list[dict] | Each has candidate_id, name, skills, experience, education |
| step | int | Current step number |

---

## Action Space

| Field | Type | Description |
|---|---|---|
| job_description | dict | Role, must-have skills (weighted), nice-to-have skills, min experience |
| candidates | list[dict] | Each has candidate_id, name, skills, experience, education |
| step | int | Current step number |

---

## Tasks

### Easy — 10 candidates
- Direct skill matching
- Explicit requirements
- No ambiguity

### Medium — 25 candidates
- Weighted skills
- Transferable skill equivalences
- Requires reasoning beyond direct matches

### Hard — 50 candidates
- Complex requirements
- Multiple constraints
- Includes uncertainty flagging

---

## Reward Function

The reward is computed based on how well the predicted ranking matches
the ground truth ranking for a given job and candidate set.

| Component | Value |
|---|---|
| Base score | Spearman rank correlation vs ground truth (0.0 – 1.0) |
| Wrong top candidate | -0.2 penalty |
| Best candidate not in top 3 | -0.2 penalty |
| Flag quality bonus (hard) | +0.1 for correctly flagged borderline candidates |
| Incorrect flag penalty (hard) | penalty for unnecessary or incorrect flags |

---


## Baseline Performance (Qwen/Qwen2.5-7B-Instruct via Hugging Face)

| Task   | Reward |
|--------|--------|
| easy   | ~0.80 – 0.90 |
| medium | ~0.70 – 0.80 |
| hard   | ~0.40 – 0.60 |

---

## Setup & Usage

### Local
```bash
git clone https://huggingface.co/spaces/Team-Duality/resume-screening-env 
cd resume-screening-env
pip install -r requirements.txt

export HF_TOKEN="token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

python inference.py
```

### Docker
```bash
docker build -t resume-screening-env .
docker run -e HF_TOKEN=token \
           -e API_BASE_URL=https://router.huggingface.co/v1 \
           -e MODEL_NAME=Qwen/Qwen2.5-7B-Instruct \
           resume-screening-env
```

## Environment Variables

| Variable | Default | Required | Description |
|---|---|---|---|
| HF_TOKEN | — | yes | Hugging Face API token (must be set in Secrets) |
| API_BASE_URL | https://router.huggingface.co/v1 | no | Base URL for OpenAI-compatible API |
| MODEL_NAME | Qwen/Qwen2.5-7B-Instruct | no | Model used for inference |

---

## Project Structure
```
resume-screening-env/
├── data/
│   ├── job_descriptions.json   # 3 job descriptions (easy/medium/hard)
│   └── resumes.json            # 50 synthetic candidate resumes
├── env/
│   ├── environment.py          # Main OpenEnv interface
│   ├── grader.py               # Ground truth scoring and rank correlation
│   ├── reward.py               # Reward shaping based on ranking accuracy
│   └── tasks.py                # Task loaders
├── models/
│   ├── action.py               # Pydantic Action model
│   ├── observation.py          # Pydantic Observation model
│   └── reward.py               # Pydantic Reward model
├── server/
│   └── app.py                  # FastAPI REST endpoints for OpenEnv validation
├── inference.py                # LLM-based agent using OpenAI-compatible HF router
├── openenv.yaml                # OpenEnv metadata
├── pyproject.toml              # Python package config for openenv validate
├── Dockerfile                  # Container definition
└── requirements.txt
```

## Output Format
```
[START] task=easy env=resume-screening model=Qwen/Qwen2.5-7B-Instruct
[STEP] step=1 action=['c01','c02',...] reward=0.82 done=true error=null
[END] success=true steps=1 rewards=0.82
```