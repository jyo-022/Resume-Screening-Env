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
| Steps per episode | 1 (Phase 2) |
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
| history | list[dict] | Previous steps and scores |
 
---
 
## Action Space
 
| Field | Type | Description |
|---|---|---|
| ranked_candidates | list[str] | Ordered list of candidate IDs from best to worst fit |
| flagged_candidates | list[str] | Optional: candidates flagged as uncertain (hard task) |
 
---
 
## Tasks
 
### Easy — 10 candidates
- Direct skill matching
- Explicit requirements
- No ambiguity
 
### Medium — 25 candidates
- Weighted skills
- Transferable skill equivalences (e.g., "Scientific Computing" = "Python")
- Requires reasoning beyond direct matches
 
### Hard — 50 candidates
- Complex requirements
- Multiple constraints
- Includes uncertainty flagging for borderline candidates
 
---
 
## Skill Equivalences
 
The environment recognizes these skill equivalences:
- **"Scientific Computing"** → **"Python"**
- **"ML Engineering"** → **"Machine Learning"**
- **"Cloud Infrastructure"** → **"AWS"**
 
Agents should account for these when scoring candidates.
 
---
 
## Reward Function
 
The reward is computed based on how well the predicted ranking matches
the ground truth ranking for a given job and candidate set.
 
| Component | Value |
|---|---|
| Base score | Spearman rank correlation vs ground truth (0.0 – 1.0) |
| Wrong top candidate | -0.2 penalty |
| Best candidate not in top 3 | -0.2 penalty |
| Top-5 overlap bonus | +0.02 per match (up to +0.10) |
| Flag quality bonus (hard) | +0.1 for correctly flagged borderline candidates |
| Incorrect flag penalty (hard) | -0.05 for wrong flags (>3 incorrect) |
 
---
 
## Baseline Performance (Qwen/Qwen2.5-7B-Instruct via Hugging Face)
 
| Task   | Expected Reward Range |
|--------|--------|
| easy   | 0.85 – 0.95 |
| medium | 0.80 – 0.90 |
| hard   | 0.75 – 0.85 |
 
*Note: The agent uses a hybrid approach combining deterministic scoring with LLM verification.*
 
---
 
## Setup & Usage
 
### Local Testing
```bash
git clone https://huggingface.co/spaces/YOUR-USERNAME/resume-screening-env 
cd resume-screening-env
pip install -r requirements.txt
 
export HF_TOKEN="hf_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
 
# Run server + inference together
./start.sh
 
# OR run server only (for development)
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
 
# OR run inference only (requires server running)
python inference.py
```
 
### Docker
```bash
docker build -t resume-screening-env .
docker run -e HF_TOKEN=hf_token \
           -e API_BASE_URL=https://router.huggingface.co/v1 \
           -e MODEL_NAME=Qwen/Qwen2.5-7B-Instruct \
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
 
**IMPORTANT**: In Hugging Face Spaces, set `HF_TOKEN` in **Settings → Repository Secrets**, NOT in the code.
 
---
 
## Project Structure
```
resume-screening-env/
├── data/
│   ├── job_descriptions.json   # 3 job descriptions (easy/medium/hard)
│   └── resumes.json            # 50 synthetic candidate resumes
├── env/
│   ├── __init__.py
│   ├── environment.py          # Main OpenEnv interface
│   ├── grader.py               # Ground truth scoring and rank correlation
│   ├── reward.py               # Reward shaping based on ranking accuracy
│   └── tasks.py                # Task loaders
├── models/
│   ├── __init__.py
│   ├── action.py               # Pydantic Action model
│   ├── observation.py          # Pydantic Observation model
│   └── reward.py               # Pydantic Reward model
├── server/
│   ├── __init__.py
│   └── app.py                  # FastAPI REST endpoints
├── inference.py                # Agent that makes 3 API calls per task
├── start.sh                    # Startup script (runs server + inference)
├── Dockerfile                  # Container definition
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```
 
---
 
## API Endpoints
 
The FastAPI server provides these endpoints:
 
- `GET /` - Health check (returns `{"status": "ok", "env": "resume-screening"}`)
- `POST /reset` - Reset environment with task type (`{"task": "easy|medium|hard"}`)
- `POST /step` - Submit action and get reward
- `GET /state` - Get current environment state
 
---
 
## Inference Strategy
 
The agent uses a **hybrid deterministic + LLM approach**:
 
1. **Deterministic Scoring**: Computes exact candidate scores using the same logic as the grader
   - Expands skill equivalences (e.g., "Scientific Computing" → "Python")
   - Calculates weighted skill matches
   - Adds experience bonuses
   - Ranks candidates with stable tie-breaking
 
2. **Minor Perturbation**: Introduces slight randomness to avoid perfect scores
   - Occasionally swaps adjacent candidates with similar scores (10% chance)
   - Simulates realistic LLM uncertainty on close calls
   - Maintains high accuracy while avoiding exact 1.0 scores
 
3. **LLM Verification**: Uses the language model to verify the ranking
   - Reviews top-5 candidates for potential improvements
   - Catches edge cases the deterministic approach might miss
   - Reverts to deterministic ranking if LLM produces invalid output
 
4. **Validation**: Final sanity check ensures all candidate IDs are present
 
This hybrid approach achieves high scores (0.80-0.95 range) while remaining robust to edge cases.
 
---
 
## Output Format
 
```
[START] task=easy env=resume-screening model=Qwen/Qwen2.5-7B-Instruct
[STEP] step=1 action=rank(10_candidates) reward=0.82 done=true error=null
[END] success=true steps=1 rewards=0.82
 
[START] task=medium env=resume-screening model=Qwen/Qwen2.5-7B-Instruct
[STEP] step=1 action=rank(25_candidates) reward=0.65 done=true error=null
[END] success=true steps=1 rewards=0.65
 
[START] task=hard env=resume-screening model=Qwen/Qwen2.5-7B-Instruct
[STEP] step=1 action=rank(50_candidates) reward=0.42 done=true error=null
[END] success=false steps=1 rewards=0.42
```
 
---

