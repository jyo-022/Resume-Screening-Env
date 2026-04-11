---
title: Resume Screening Env
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
app_file: start.sh
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
| Steps per episode | 1 |
| Reward range | 0.01 – 0.99 |
| Action type | Ranked list of candidate IDs |
| Observation type | Job description + candidate pool |

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| job_description | dict | Role, must-have skills (weighted), nice-to-have skills, min experience |
| candidates | list[dict] | Each has candidate_id, name, skills, experience, education |
| step | int | Current step number |
| history | list[dict] | Previous steps and scores |

---

## Action Space

| Field | Type | Description |
|---|---|---|
| ranked_candidates | list[str] | Ordered list of candidate IDs from best to worst fit |
| flagged_candidates | list[str] | Optional: candidates flagged as uncertain (hard task only) |

---

## Tasks

### Easy — 10 candidates
- Direct skill matching against explicit must-have requirements
- No ambiguity, no transferable skills
- Graded purely on Spearman rank correlation against ground truth

### Medium — 25 candidates
- Weighted must-have and nice-to-have skills
- Transferable skill equivalences (e.g., "Scientific Computing" = "Python")
- Requires reasoning beyond direct keyword matches

### Hard — 50 candidates
- Complex multi-constraint requirements
- Agent must rank all 50 candidates accurately
- Includes uncertainty flagging: agent should flag borderline candidates for human review
- Graded on rank correlation plus quality of uncertainty flags

---

## Skill Equivalences

The environment recognizes these transferable skill mappings:

| Candidate Skill | Counts As |
|---|---|
| Scientific Computing | Python |
| ML Engineering | Machine Learning |
| Cloud Infrastructure | AWS |

Agents that correctly identify these equivalences will score higher than those doing pure keyword matching.

---

## Reward Function

The reward is computed based on how well the predicted ranking matches
the ground truth ranking produced by a deterministic scoring rubric.

| Component | Value |
|---|---|
| Base score | Spearman rank correlation vs ground truth, capped at 0.90 |
| Wrong top candidate | -0.05 penalty |
| Best candidate not in top 3 | -0.05 penalty |
| Top-5 overlap bonus | +0.01 per match (up to +0.05) |
| Flag quality bonus (hard only) | +0.05 for correctly flagged borderline candidates |
| Incorrect flag penalty (hard only) | -0.03 for >3 wrong flags |
| Difficulty offset (easy) | -0.02 fixed |
| Difficulty offset (medium) | -0.05 fixed |
| Difficulty offset (hard) | -0.10 fixed |
| Score range (enforced) | Clamped to [0.01, 0.99] |

### Ground Truth Scoring Rubric

For each candidate, the rubric computes:
```
score = sum(weight for each must-have skill present)
      + sum(weight for each nice-to-have skill present)
      + 2 (if experience >= minimum required)
```
Candidates are ranked highest-to-lowest by this score. Ties broken alphabetically by candidate ID.

---

## Baseline Performance

Evaluated using `Qwen/Qwen2.5-7B-Instruct` via Hugging Face Inference API.
Agent uses a hybrid deterministic + LLM verification approach.

| Task | Candidates | Baseline Score | Success |
|------|------------|---------------|---------|
| easy | 10 | ~0.85 | ✅ |
| medium | 25 | ~0.75 | ✅ |
| hard | 50 | ~0.62 | ✅ |

*Scores are deterministic and reproducible across runs.*

---

## Setup & Usage

### Local Testing
```bash
git clone https://huggingface.co/spaces/Team-Duality/resume-screening-env
cd resume-screening-env
pip install -r requirements.txt

export HF_TOKEN="your_hf_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

# Run server + inference together
./start.sh

# OR run server only (for development)
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860

# OR run inference only (requires server already running)
python inference.py
```

### Docker
```bash
docker build -t resume-screening-env .
docker run -e HF_TOKEN=your_hf_token \
           -e API_BASE_URL=https://router.huggingface.co/v1 \
           -e MODEL_NAME=Qwen/Qwen2.5-7B-Instruct \
           -p 7860:7860 \
           resume-screening-env
```

---

## Environment Variables

| Variable | Default | Required | Description |
|---|---|---|---|
| HF_TOKEN | — | **YES** | Hugging Face API token (set in Space Secrets) |
| API_BASE_URL | https://router.huggingface.co/v1 | no | Base URL for OpenAI-compatible API |
| MODEL_NAME | Qwen/Qwen2.5-7B-Instruct | no | Model used for inference |
| ENV_URL | http://127.0.0.1:7860 | no | Environment server URL (auto-set by validator) |

**IMPORTANT**: In Hugging Face Spaces, set `HF_TOKEN` in **Settings → Repository Secrets**, NOT hardcoded in the code.

---

## Project Structure

```
resume-screening-env/
├── data/
│   ├── job_descriptions.json   # 3 job descriptions (easy/medium/hard)
│   └── resumes.json            # 50 synthetic candidate resumes
├── env/
│   ├── __init__.py
│   ├── environment.py          # Main OpenEnv interface (reset/step/state)
│   ├── grader.py               # Ground truth scoring and Spearman correlation
│   ├── reward.py               # Reward shaping with deterministic bonuses/penalties
│   └── tasks.py                # Task loaders (easy=10, medium=25, hard=50 candidates)
├── models/
│   ├── __init__.py
│   ├── action.py               # Pydantic Action model
│   ├── observation.py          # Pydantic Observation model
│   └── reward.py               # Pydantic Reward model
├── server/
│   ├── __init__.py
│   └── app.py                  # FastAPI REST endpoints
├── inference.py                # Baseline agent (3 tasks × 1 LLM call each)
├── openenv.yaml                # OpenEnv spec metadata
├── start.sh                    # Startup script (server + inference)
├── Dockerfile                  # Container definition
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check — returns `{"status": "ok", "env": "resume-screening"}` |
| POST | `/reset` | Reset environment — body: `{"task": "easy\|medium\|hard"}` |
| POST | `/step` | Submit action — body: `{"ranked_candidates": [...], "flagged_candidates": [...]}` |
| GET | `/state` | Get current environment state |

---

## Inference Strategy

The agent uses a **hybrid deterministic + LLM verification** approach:

1. **Deterministic Scoring**: Computes exact candidate scores using the same logic as the grader
   - Expands skill equivalences (e.g., "Scientific Computing" → "Python")
   - Calculates weighted skill matches for must-have and nice-to-have requirements
   - Adds experience bonuses for candidates meeting the minimum threshold
   - Ranks candidates with stable tie-breaking (score descending, ID ascending)

2. **LLM Verification**: Uses the language model to verify and potentially refine the ranking
   - Presents all candidates with scoring rules explicitly in the prompt
   - Catches edge cases the deterministic approach might miss
   - Reverts to deterministic ranking if LLM produces invalid or incomplete output

3. **Validation**: Final sanity check ensures all candidate IDs are present before submitting

This approach is fully deterministic and reproducible — no randomness is introduced at any stage.

---

## Output Format

```
[START] task=easy env=resume-screening model=Qwen/Qwen2.5-7B-Instruct
[STEP] step=1 action=rank(10_candidates) reward=0.85 done=true error=null
[END] success=true steps=1 score=0.85 rewards=0.85

[START] task=medium env=resume-screening model=Qwen/Qwen2.5-7B-Instruct
[STEP] step=1 action=rank(25_candidates) reward=0.75 done=true error=null
[END] success=true steps=1 score=0.75 rewards=0.75

[START] task=hard env=resume-screening model=Qwen/Qwen2.5-7B-Instruct
[STEP] step=1 action=rank(50_candidates) reward=0.62 done=true error=null
[END] success=true steps=1 score=0.62 rewards=0.62
```

---

## Grading Details

| Task | Grader | Metric |
|---|---|---|
| easy | Spearman rank correlation | Score in [0.01, 0.99] |
| medium | Spearman + transferable skill credit | Score in [0.01, 0.99] |
| hard | Spearman + uncertainty flag quality | Score in [0.01, 0.99] |

All graders are deterministic and reproducible. Scores are never exactly 0.0 or 1.0.

---

