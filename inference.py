import os
import json
import time
from openai import OpenAI
 


# ── Environment variables ──────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN", "")
# ENV_URL: Validator sets this to container URL, defaults to localhost for local testing
ENV_URL      = os.getenv("ENV_URL", "http://127.0.0.1:7860")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

client   = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
ENV_NAME = "resume-screening"

# ── Prompt builder ─────────────────────────────────────────────────
def build_prompt(job, candidates, iteration_note=""):
    must = ", ".join(f"{r['skill']}(weight={r['weight']})" for r in job["must_have"])
    nice = ", ".join(f"{r['skill']}(weight={r['weight']})" for r in job["nice_to_have"])

    lines = [
        "You are an expert recruiter.",
        iteration_note,
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
        "  - CRITICAL EQUIVALENCES:",
        "    * 'Scientific Computing' = Python",
        "    * 'ML Engineering' = Machine Learning",
        "    * 'Cloud Infrastructure' = AWS",
        "",
        "Candidates:",
    ]

    for c in candidates:
        lines.append(
            f"  ID={c['candidate_id']} | Skills={c['skills']} | "
            f"Experience={c['experience']}yr | Education={c['education']}"
        )

    all_ids = [c["candidate_id"] for c in candidates]
    ids_str = json.dumps(all_ids)

    lines += [
        "",
        "Instructions:",
        "1. Calculate total points for each candidate",
        "2. Rank from highest to lowest score",
        f"3. Use ONLY these exact {len(all_ids)} candidate IDs: {ids_str}",
        "",
        f"Return ONLY this JSON with all {len(all_ids)} IDs:",
        '{"ranked_candidates": ["c01", "c02", ...], "flagged_candidates": []}',
        "No markdown, no explanation, just the JSON object.",
    ]

    return "\n".join(lines)

# ── Parse LLM response ─────────────────────────────────────────────
def parse_action(text, all_ids):
    try:
        text = text.strip()
        
        # Remove markdown code blocks
        if "```" in text:
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else parts[0]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        # Remove quotes
        if text.startswith(("'", '"')) and text.endswith(("'", '"')):
            text = text[1:-1].strip()

        data = json.loads(text)
        ranked = data.get("ranked_candidates", [])
        flagged = data.get("flagged_candidates", [])

        # Validate and deduplicate
        seen = set()
        valid_ranked = []
        for c in ranked:
            if c in all_ids and c not in seen:
                valid_ranked.append(c)
                seen.add(c)

        # Add missing candidates
        missing = [c for c in all_ids if c not in seen]
        valid_ranked.extend(missing)

        # Validate flagged candidates
        valid_flagged = [c for c in flagged if c in valid_ranked]
        
        return {
            "ranked_candidates": valid_ranked,
            "flagged_candidates": valid_flagged
        }

    except Exception as e:
        # Fallback: return all IDs in original order
        return {
            "ranked_candidates": list(all_ids),
            "flagged_candidates": []
        }

# ── Run one episode with 3 API calls ───────────────────────────────
def run_task(task_name):
    """Run inference on a single task with exactly 3 API calls"""
    import requests
    
    # Reset environment
    try:
        resp = requests.post(f"{ENV_URL}/reset", json={"task": task_name}, timeout=10)
        resp.raise_for_status()
        obs = resp.json()
    except Exception as e:
        print(f"[ERROR] Failed to reset environment: {e}", flush=True)
        return

    job = obs["job_description"]
    candidates = obs["candidates"]
    all_ids = [c["candidate_id"] for c in candidates]
    
    print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    last_error = "null"
    
    try:
        # ── API Call 1: Initial ranking ────────────────────────────
        prompt1 = build_prompt(job, candidates, 
            iteration_note="ITERATION 1/3: Provide your initial ranking based on the scoring guide.")
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt1}],
                temperature=0.0,
            )
            raw1 = response.choices[0].message.content
            action_1 = parse_action(raw1, all_ids)
        except Exception as e:
            action_1 = {"ranked_candidates": list(all_ids), "flagged_candidates": []}
            last_error = str(e).replace("\n", " ")[:120]

        # ── API Call 2: Refinement ─────────────────────────────────
        top3 = action_1["ranked_candidates"][:3]
        prompt2 = build_prompt(job, candidates,
            iteration_note=f"ITERATION 2/3: Your first ranking had {top3} in top 3. Verify skill equivalences and recalculate scores.")
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt2}],
                temperature=0.0,
            )
            raw2 = response.choices[0].message.content
            action_2 = parse_action(raw2, all_ids)
        except Exception as e:
            action_2 = action_1
            last_error = str(e).replace("\n", " ")[:120]

        # ── API Call 3: Final validation ───────────────────────────
        top3 = action_2["ranked_candidates"][:3]
        prompt3 = build_prompt(job, candidates,
            iteration_note=f"ITERATION 3/3: Final check. Current top 3: {top3}. Ensure all {len(all_ids)} IDs included and correctly ranked.")
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt3}],
                temperature=0.0,
            )
            raw3 = response.choices[0].message.content
            final_action = parse_action(raw3, all_ids)
        except Exception as e:
            final_action = action_2
            last_error = str(e).replace("\n", " ")[:120]

        # ── Submit final action to environment ─────────────────────
        try:
            resp = requests.post(f"{ENV_URL}/step", json=final_action, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            
            reward = result["reward"]
            done = result["done"]
            error = result.get("info", {}).get("error", "null")
            
            action_str = f"rank({len(final_action['ranked_candidates'])}_candidates)"
            error_str = error if error != "null" else last_error
            
            print(
                f"[STEP] step=1 "
                f"action={action_str} "
                f"reward={reward:.2f} "
                f"done={'true' if done else 'false'} "
                f"error={error_str}",
                flush=True
            )
            
            success = reward > 0.5
            print(
                f"[END] success={'true' if success else 'false'} "
                f"steps=1 rewards={reward:.2f}",
                flush=True
            )
            
        except Exception as e:
            error_msg = str(e).replace("\n", " ")[:120]
            print(
                f"[STEP] step=1 action=error "
                f"reward=0.00 done=true error={error_msg}",
                flush=True
            )
            print(
                f"[END] success=false steps=1 rewards=0.00",
                flush=True
            )

    except Exception as e:
        error_msg = str(e).replace("\n", " ")[:120]
        print(
            f"[STEP] step=1 action=error "
            f"reward=0.00 done=true error={error_msg}",
            flush=True
        )
        print(
            f"[END] success=false steps=1 rewards=0.00",
            flush=True
        )

# ── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[INFO] Connecting to environment at {ENV_URL}", flush=True)
    
    # Run all three tasks
    for task in ["easy", "medium", "hard"]:
        run_task(task)
        print(flush=True)
        time.sleep(1)
    
    print("[INFO] Inference complete.", flush=True)
