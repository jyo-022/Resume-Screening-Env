SKILL_EQUIVALENCES = {
    "Scientific Computing": "Python",
    "ML Engineering":       "Machine Learning",
    "Cloud Infrastructure": "AWS",
}

def compute_ground_truth_scores(job, candidates):
    scores = {}
    for candidate in candidates:
        expanded_skills = set(candidate["skills"])
        for skill in list(expanded_skills):
            if skill in SKILL_EQUIVALENCES:
                expanded_skills.add(SKILL_EQUIVALENCES[skill])

        score = 0
        for req in job["must_have"]:
            if req["skill"] in expanded_skills:
                score += req["weight"]
        for req in job["nice_to_have"]:
            if req["skill"] in expanded_skills:
                score += req["weight"]
        if candidate["experience"] >= job["min_experience"]:
            score += 2

        scores[candidate["candidate_id"]] = score
    return scores

def rank_candidates(scores):
    return sorted(scores, key=scores.get, reverse=True)

def compute_rank_correlation(predicted, actual):
    n = len(actual)
    if n <= 1:
        return 1.0

    pred_ranks = {c: i for i, c in enumerate(predicted)}
    actual_ranks = {c: i for i, c in enumerate(actual)}

    d_squared_sum = 0
    for c in actual:
        d = pred_ranks[c] - actual_ranks[c]
        d_squared_sum += d * d

    score = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
    return max(0.0, min(1.0, score))

def get_borderline_candidates(scores, margin=2):
    if not scores:
        return []
    score_values = sorted(scores.values(), reverse=True)
    n = len(score_values)
    boundary_score = score_values[n // 2]
    borderline = [
        cid for cid, s in scores.items()
        if abs(s - boundary_score) <= margin
    ]
    return borderline