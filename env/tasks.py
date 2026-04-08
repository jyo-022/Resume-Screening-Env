import json
import copy

def _load_all():
    with open("data/job_descriptions.json") as f:
        jobs = json.load(f)
    with open("data/resumes.json") as f:
        candidates = json.load(f)
    return jobs, candidates

def load_easy_task():
    jobs, all_candidates = _load_all()
    job = jobs[0]
    candidates = copy.deepcopy(all_candidates[:10])  # FIXED: Exactly 10
    return job, candidates

def load_medium_task():
    jobs, all_candidates = _load_all()
    job = jobs[1] 
    candidates = copy.deepcopy(all_candidates[:25])  # Exactly 25
    return job, candidates

def load_hard_task():
    jobs, all_candidates = _load_all()
    job = copy.deepcopy(jobs[2])
    candidates = copy.deepcopy(all_candidates[:50])  # FIXED: Exactly 50
    return job, candidates