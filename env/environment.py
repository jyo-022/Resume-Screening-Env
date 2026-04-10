from env.tasks import load_easy_task, load_medium_task, load_hard_task
from env.reward import compute_final_reward
from models.observation import Observation
from models.action import Action
from models.reward import Reward
from env.grader import compute_ground_truth_scores, rank_candidates, compute_rank_correlation

class ResumeScreeningEnv:
    def __init__(self, task="easy"):
        self.task = task
        self.current_step = 0
        self.done = False
        self.history = []

        # SINGLE STEP FOR ALL TASKS (PHASE 2)
        self.max_steps = 1
        
        if task == "easy":
            self.job, self.candidates = load_easy_task()
        elif task == "medium":
            self.job, self.candidates = load_medium_task()
        elif task == "hard":
            self.job, self.candidates = load_hard_task()
        else:
            raise ValueError("Invalid task type")

        # Ground truth
        self.gt_scores = compute_ground_truth_scores(self.job, self.candidates)
        self.gt_ranking = rank_candidates(self.gt_scores)

    def reset(self):
        self.current_step = 0
        self.done = False
        self.history = []
        return Observation(
            job_description=self.job,
            candidates=self.candidates,
            step=self.current_step,
            history=[]
        )

    def step(self, action: Action):
        self.current_step = 1
        
        candidate_ids = [c["candidate_id"] for c in self.candidates]
        n_expected = len(candidate_ids)

        # Epsilon for strict (0, 1) range
        epsilon = 1e-6

        # LEN CHECK FIRST (Hard task fix)
        if len(action.ranked_candidates) != n_expected:
            score = 0.1  # Changed from 0.0 to strictly > 0
            error = f"Wrong length: got {len(action.ranked_candidates)}, expected {n_expected}"
            reward = Reward(score=score)
            return self._obs(), reward, True, {"error": error}

        # SET VALIDATION (relaxed - partial credit)
        ranked_set = set(action.ranked_candidates)
        expected_set = set(candidate_ids)
        missing = expected_set - ranked_set
        extra = ranked_set - expected_set
        
        if missing or extra:
            score = 0.15  # Partial credit, strictly > 0
            error = f"Missing {len(missing)}, extra {len(extra)}"
            reward = Reward(score=score)
            return self._obs(), reward, True, {"error": error}

        # FULL SCORING (perfect ranking)
        base_score = compute_rank_correlation(action.ranked_candidates, self.gt_ranking)
        final_score = compute_final_reward(
            base_score=base_score,
            predicted=action.ranked_candidates,
            ground_truth=self.gt_ranking,
            flagged=action.flagged_candidates,
            gt_scores=self.gt_scores,
            task=self.task,
            step=1,
            max_steps=1,
            history=self.history
        )

        # CRITICAL: Ensure score is strictly in (0, 1) range
        final_score = max(epsilon, min(1.0 - epsilon, float(final_score)))

        reward = Reward(score=float(final_score))
        self.done = True
        
        self.history.append({
            "step": 1,
            "ranked_candidates": action.ranked_candidates,
            "score": float(final_score)
        })

        return self._obs(), reward, True, {"error": "null"}

    def _obs(self):
        return Observation(
            job_description=self.job,
            candidates=self.candidates,
            step=self.current_step,
            history=self.history
        )

    def state(self):
        return {
            "task": self.task,
            "step": self.current_step,
            "done": self.done,
            "max_steps": self.max_steps,
            "num_candidates": len(self.candidates),
            "history": self.history
        }

    def close(self):
        pass