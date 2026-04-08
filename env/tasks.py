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
    job = jobs[0]                          # job_easy
    candidates = copy.deepcopy(all_candidates[:10])   # first 10 candidates
    return job, candidates


def load_medium_task():
    jobs, all_candidates = _load_all()
    job = jobs[1]                          # job_medium
    candidates = copy.deepcopy(all_candidates[:25])   # first 25 candidates
    return job, candidates


def load_hard_task():
    jobs, all_candidates = _load_all()
    job = copy.deepcopy(jobs[2])           # job_hard — deep copy so mutation is safe
    candidates = copy.deepcopy(all_candidates)        # all 50 candidates
    return job, candidates