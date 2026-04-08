import os
import json
import threading
from openai import OpenAI
from env.environment import ResumeScreeningEnv
from models.action import Action

# ── Environment variables ──────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", HF_TOKEN)
client   = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
ENV_NAME = "resume-screening"


# ── Health server (keeps HF Space alive) ──────────────────────────
def start_health_server():
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn

    health_app = FastAPI()
    env_store  = {}

    class TaskRequest(BaseModel):
        task: str = "easy"

    @health_app.get("/")
    def health():
        return {"status": "ok", "env": ENV_NAME}

    @health_app.post("/reset")
    def reset(req: TaskRequest = None):
        task = req.task if req else "easy"
        env  = ResumeScreeningEnv(task=task)
        obs  = env.reset()
        env_store["current"] = env
        return {
            "job_description": obs.job_description,
            "candidates":      obs.candidates,
            "step":            obs.step,
            "history":         obs.history,
        }

    @health_app.post("/step")
    def step(act: Action):
        env = env_store.get("current")
        if not env:
            return {"error": "call /reset first"}
        obs, reward, done, info = env.step(act)        # ← correctly indented inside step()
        return {                                        # ← correctly indented inside step()
            "observation": {
                "job_description": obs.job_description,
                "candidates":      obs.candidates,
                "step":            obs.step,
                "history":         obs.history,
            },
            "reward": reward.score,
            "done":   done,
            "info":   info,
        }

    @health_app.get("/state")
    def state():
        env = env_store.get("current")
        if not env:
            return {"error": "call /reset first"}
        return env.state()

    try:
        uvicorn.run(health_app, host="0.0.0.0", port=7860, log_level="error")
    except Exception:
        pass


# Start health server as daemon thread immediately
_server_thread = threading.Thread(target=start_health_server, daemon=True)
_server_thread.start()


# ── Prompt builder ─────────────────────────────────────────────────
def build_prompt(observation, step, max_steps):
    job  = observation.job_description
    must = ", ".join(f"{r['skill']}(weight={r['weight']})" for r in job["must_have"])
    nice = ", ".join(f"{r['skill']}(weight={r['weight']})" for r in job["nice_to_have"])

    prev_ranking = []
    prev_score   = None
    if observation.history:
        prev_ranking = observation.history[-1]["ranked_candidates"]
        prev_score   = observation.history[-1]["score"]

    lines = [
        f"You are an expert recruiter. This is step {step} of {max_steps}.",
        "",
        f"Role: {job['role']}",
        f"Must-have skills (required): {must}",
        f"Nice-to-have skills (bonus): {nice}",
        f"Minimum experience: {job['min_experience']} years",
        "",
        "Scoring guide (rank by total points):",
        "  - Each must-have skill matched = its weight in points",
        "  - Each nice-to-have skill matched = its weight in points",
        "  - Meeting minimum experience = +2 points",
        "  - Note: 'Scientific Computing' counts as Python",
        "",
        "Candidates:",
    ]

    for c in observation.candidates:
        lines.append(
            f"  ID={c['candidate_id']} | Skills={c['skills']} | "
            f"Experience={c['experience']}yr | Education={c['education']}"
        )

    if prev_ranking:
        lines += [
            "",
            f"Your previous ranking (step {step - 1}, score={prev_score:.2f}):",
            f"  Top-5: {prev_ranking[:5]}",
            f"  Bottom-5: {prev_ranking[-5:]}",
            "",
            "Carefully reconsider and improve this ranking.",
        ]
    else:
        lines += [
            "",
            "Carefully score each candidate using the guide above.",
            "Rank from highest score to lowest score.",
        ]

    all_ids = [c["candidate_id"] for c in observation.candidates]
    ids_str = json.dumps(all_ids)

    lines += [
        "",
        f"IMPORTANT: Use ONLY these exact {len(all_ids)} candidate IDs:",
        f"  {ids_str}",
        "",
        f"Return ONLY this JSON with all {len(all_ids)} IDs included:",
        '{"ranked_candidates": ["c01", "c02", ...], "flagged_candidates": []}',
        "No markdown, no explanation, just the JSON object.",
    ]

    return "\n".join(lines)


# ── Parse LLM response ─────────────────────────────────────────────
def parse_action(text, all_ids):
    text = text.strip()

    if "```" in text:
        parts = text.split("```")
        text  = parts[1] if len(parts) > 1 else parts[0]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    if text.startswith(("'", '"')) and text.endswith(("'", '"')):
        text = text[1:-1].strip()

    data    = json.loads(text)
    ranked  = data.get("ranked_candidates", [])
    flagged = data.get("flagged_candidates", [])

    valid_ranked  = [cid for cid in ranked if cid in all_ids]
    missing       = [cid for cid in all_ids if cid not in valid_ranked]
    if missing:
        valid_ranked.extend(missing)

    valid_flagged = [cid for cid in flagged if cid in valid_ranked]

    return Action(
        ranked_candidates=valid_ranked,
        flagged_candidates=valid_flagged
    )


# ── Run one episode ────────────────────────────────────────────────
def run_task(task_name):
    env        = ResumeScreeningEnv(task=task_name)
    obs        = env.reset()
    all_ids    = [c["candidate_id"] for c in obs.candidates]
    rewards    = []
    step_count = 0
    done       = False
    success    = False
    last_error = "null"

    # 1 LLM call per task by default to stay within API budget.
    # Set INFERENCE_MAX_STEPS=3 env var to enable full multi-step mode.
    inference_max_steps = int(os.getenv("INFERENCE_MAX_STEPS", "1"))

    print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    try:
        while not done and step_count < inference_max_steps:
            step_count += 1
            prompt = build_prompt(obs, step=step_count, max_steps=env.max_steps)

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                raw        = response.choices[0].message.content
                action     = parse_action(raw, all_ids)
                last_error = "null"
            except Exception as e:
                action     = Action(ranked_candidates=list(all_ids))
                last_error = str(e).replace("\n", " ")[:120]

            obs, reward, done, info = env.step(action)
            rewards.append(reward.score)

            action_str = str(action.ranked_candidates)
            error_str  = info.get("error") or last_error or "null"

            print(
                f"[STEP] step={step_count} "
                f"action={action_str} "
                f"reward={reward.score:.2f} "
                f"done={'true' if done else 'false'} "
                f"error={error_str}",
                flush=True
            )

        success = bool(rewards) and rewards[-1] > 0.5

    except Exception as e:
        last_error = str(e).replace("\n", " ")[:120]
        print(
            f"[STEP] step={max(step_count, 1)} action=error "
            f"reward=0.00 done=true error={last_error}",
            flush=True
        )
        rewards.append(0.0)
        success = False

    finally:
        env.close()
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={'true' if success else 'false'} "
            f"steps={step_count} rewards={rewards_str}",
            flush=True
        )


# ── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)
        print(flush=True)

    # keep main thread alive so daemon health server stays running
    _server_thread.join()