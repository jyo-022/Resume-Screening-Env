def compute_final_reward(base_score, predicted, ground_truth,
                         flagged=None, gt_scores=None, task="easy",
                         step=1, max_steps=3, history=None):

    reward = max(0.0, min(1.0, base_score))

    # ── Step-based shaping ─────────────────────────────────────────
    # Early steps: reward partial progress, be lenient on penalties
    # Final step: full penalties apply
    is_final_step = (step >= max_steps)

    # Reward improvement over previous step
    if history and len(history) > 0:
        prev_score = history[-1]["score"]
        improvement = reward - prev_score
        if improvement > 0:
            # bonus for getting better each step
            reward += 0.05
        elif improvement < -0.05:
            # penalty for getting significantly worse
            reward -= 0.05

    # ── Ranking quality penalties (full weight only on final step) ─
    penalty_weight = 1.0 if is_final_step else 0.5

    if predicted[0] != ground_truth[0]:
        reward -= 0.2 * penalty_weight

    if ground_truth[0] not in predicted[:3]:
        reward -= 0.2 * penalty_weight

    # ── Top-5 overlap bonus ────────────────────────────────────────
    predicted_top5 = set(predicted[:5])
    gt_top5 = set(ground_truth[:5])
    overlap = len(predicted_top5 & gt_top5)
    reward += overlap * 0.02  # up to +0.10 bonus for top-5 matches

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
                reward += 0.1
            wrong_flags = flagged_set - true_borderline
            if len(wrong_flags) > 3:
                reward -= 0.05

    # ── CRITICAL: Clamp to strictly (0, 1) exclusive range ─────────
    # Validator requires scores strictly between 0 and 1 (not inclusive)
    epsilon = 1e-6
    reward = max(epsilon, min(1.0 - epsilon, reward))
    
    return reward