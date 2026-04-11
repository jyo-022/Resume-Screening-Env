def clamp_validator_score(x: float) -> float:
    """Clamp to (0.01, 0.99) so :.2f never prints 0.00 or 1.00."""
    x = float(x)
    if x >= 0.995:
        return 0.99
    if x <= 0.005:
        return 0.01
    return round(x, 2)


def compute_final_reward(base_score, predicted, ground_truth,
                         flagged=None, gt_scores=None, task="easy",
                         step=1, max_steps=3, history=None):
    """
    Compute final reward from base Spearman correlation score.
    base_score is already clamped to (0.01, 0.90) by grader.py.
    All bonuses/penalties are deterministic (no random).
    """
    reward = float(base_score)

    is_final_step = (step >= max_steps)

    # ── Step improvement bonus ─────────────────────────────────────
    if history and len(history) > 0:
        prev_score = history[-1]["score"]
        improvement = reward - prev_score
        if improvement > 0:
            reward += 0.03
        elif improvement < -0.05:
            reward -= 0.03

    # ── Ranking quality penalties ──────────────────────────────────
    penalty_weight = 1.0 if is_final_step else 0.5

    if predicted[0] != ground_truth[0]:
        reward -= 0.05 * penalty_weight

    if ground_truth[0] not in predicted[:3]:
        reward -= 0.05 * penalty_weight

    # ── Top-5 overlap bonus (up to +0.05) ─────────────────────────
    predicted_top5 = set(predicted[:5])
    gt_top5 = set(ground_truth[:5])
    overlap = len(predicted_top5 & gt_top5)
    reward += overlap * 0.01

    # ── Hard task: flagging quality ────────────────────────────────
    if task == "hard" and flagged is not None and gt_scores is not None:
        from env.grader import get_borderline_candidates
        true_borderline = set(get_borderline_candidates(gt_scores))
        flagged_set = set(flagged)

        if true_borderline:
            correctly_flagged = len(flagged_set & true_borderline)
            flag_precision = correctly_flagged / len(flagged_set) if flagged_set else 0.0
            flag_recall = correctly_flagged / len(true_borderline)

            if flag_precision > 0.5 and flag_recall > 0.5:
                reward += 0.05
            wrong_flags = flagged_set - true_borderline
            if len(wrong_flags) > 3:
                reward -= 0.03

    # ── Deterministic task difficulty offsets ─────────────────────
    if task == "easy":
        reward -= 0.02
    elif task == "medium":
        reward -= 0.05
    elif task == "hard":
        reward -= 0.10

    # ── Display-safe clamp: :.2f will never print 0.00 or 1.00 ────
    return clamp_validator_score(reward)
