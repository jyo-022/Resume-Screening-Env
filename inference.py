import os
import json
import time
from openai import OpenAI
from env.environment import ResumeScreeningEnv
from models.action import Action

# ── Environment variables ──────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN", "")

client   = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
ENV_NAME = "resume-screening"


# ── Prompt builder ─────────────────────────────────────────────────
def build_prompt(observation):
    job  = observation.job_description
    must = ", ".join(f"{r['skill']}(weight={r['weight']})" for r in job["must_have"])
    nice = ", ".join(f"{r['skill']}(weight={r['weight']})" for r in job["nice_to_have"])

    lines = [
        "You are a recruiter. Rank ALL candidates from best to worst.",
        f"Role: {job['role']}",
        f"Must-have skills: {must}",
        f"Nice-to-have skills: {nice}",
        f"Minimum experience: {job['min_experience']} years",
        "",
        "Candidates:",
    ]
    for c in observation.candidates:
        lines.append(
            f"ID={c['candidate_id']} | Skills={c['skills']} | "
            f"Experience={c['experience']} | Education={c['education']}"
        )
    lines += [
        "",
        "Return ONLY JSON — no explanation, no markdown:",
        '{"ranked_candidates": ["c01","c02",...], "flagged_candidates": []}',
    ]
    return "\n".join(lines)


# ── Parse response ─────────────────────────────────────────────────
def parse_action(text, all_ids):
    try:
        text = text.strip()
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        if text.startswith("'") or text.startswith('"'):
            text = text[1:]
        if text.endswith("'") or text.endswith('"'):
            text = text[:-1]
        text = text.strip()

        data    = json.loads(text)
        ranked  = data.get("ranked_candidates", [])
        flagged = data.get("flagged_candidates", [])

        seen = set()
        valid_ranked = []
        for c in ranked:
            if c in all_ids and c not in seen:
                valid_ranked.append(c)
                seen.add(c)

        missing = [c for c in all_ids if c not in seen]
        valid_ranked.extend(missing)

        valid_flagged = [c for c in flagged if c in valid_ranked]
        return Action(ranked_candidates=valid_ranked, flagged_candidates=valid_flagged)

    except Exception:
        return Action(ranked_candidates=list(all_ids), flagged_candidates=[])


# ── Run one task ───────────────────────────────────────────────────
def run_task(task_name):
    env        = ResumeScreeningEnv(task=task_name)
    obs        = env.reset()
    all_ids    = [c["candidate_id"] for c in obs.candidates]
    rewards    = []
    step_count = 0
    done       = False
    success    = False
    last_error = "null"

    print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    try:
        step_count += 1
        prompt = build_prompt(obs)

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
            last_error = str(e).replace("\n", " ")[:100]

        obs, reward, done, info = env.step(action)
        rewards.append(reward.score)

        error_str = info.get("error") or last_error or "null"

        print(
            f"[STEP] step={step_count} "
            f"action=rank({len(action.ranked_candidates)}_candidates) "
            f"reward={reward.score:.2f} "
            f"done={'true' if done else 'false'} "
            f"error={error_str}",
            flush=True
        )

        success = reward.score > 0.5

    except Exception as e:
        last_error = str(e).replace("\n", " ")[:100]
        print(
            f"[STEP] step=1 action=error reward=0.00 done=true error={last_error}",
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


# ── Health server ──────────────────────────────────────────────────
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
        }

    @health_app.post("/step")
    def step(action: dict):
        env = env_store.get("current")
        if not env:
            return {"error": "call /reset first"}
        act = Action(
            ranked_candidates=action.get("ranked_candidates", []),
            flagged_candidates=action.get("flagged_candidates", [])
        )
        obs, reward, done, info = env.step(act)
        return {
            "observation": {
                "job_description": obs.job_description,
                "candidates":      obs.candidates,
                "step":            obs.step,
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

    # run_task() is already done before this is called
    # this just keeps the Space alive
    uvicorn.run(health_app, host="0.0.0.0", port=7860, log_level="error")


# ── Main — runs once, cleanly ──────────────────────────────────────
if __name__ == "__main__":
    # Step 1: run inference for all tasks (3 API calls total)
    for task in ["easy", "medium", "hard"]:
        run_task(task)
        print(flush=True)
        time.sleep(3)   # small pause between tasks to avoid rate limits

    # Step 2: start health server (blocks here, keeps Space alive)
    # inference is already done — this never triggers run_task again
    start_health_server()