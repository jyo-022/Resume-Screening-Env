import uvicorn
from fastapi import FastAPI
from env.environment import ResumeScreeningEnv
from models.action import Action

ENV_NAME  = "resume-screening"
app       = FastAPI()
env_store = {}


@app.get("/")
def health():
    return {"status": "ok", "env": ENV_NAME}


@app.post("/reset")
def reset(req: dict = None):
    task = (req or {}).get("task", "easy")
    env  = ResumeScreeningEnv(task=task)
    obs  = env.reset()
    env_store["current"] = env
    return {
        "job_description": obs.job_description,
        "candidates":      obs.candidates,
        "step":            obs.step,
        "history":         obs.history,
    }


@app.post("/step")
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
            "history":         obs.history,
        },
        "reward": reward.score,
        "done":   done,
        "info":   info,
    }


@app.get("/state")
def state():
    env = env_store.get("current")
    if not env:
        return {"error": "call /reset first"}
    return env.state()


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, log_level="info")


if __name__ == "__main__":
    main()