"""
BM25 reward manager for verl-tool.

Scores the model's extracted <answer>...</answer> against the ground-truth
string using character-level BM25Okapi (rank-bm25).  Numeric strings are
normalised before scoring so "8.0" and "8" compare as identical.

Score range: [0.0, 1.0]
  1.0 → exact match (or BM25 self-similarity == 1)
  0.0 → no character overlap

Deploy by copying this file to:
    <verl_tool_root>/workers/reward_manager/bm25_reward.py

Dependency:  pip install rank-bm25
"""

import json
import re

import torch

from verl_tool.workers.reward_manager import register

try:
    from rank_bm25 import BM25Okapi

    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_answer(response: str) -> str:
    """Return content inside the last <answer>...</answer> block, or the
    full response stripped of whitespace if no tag is present."""
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return response.strip()


def _normalize(s: str) -> str:
    """Normalise numeric strings so '8', '8.0', '8.00' all compare equally."""
    s = s.strip()
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return f"{val:.6g}"
    except ValueError:
        return s.lower()


def _bm25_score(hypothesis: str, reference: str) -> float:
    """Compute a normalised BM25 similarity in [0, 1].

    Character-level tokens are used so that partially-correct numeric
    answers (e.g. '123' vs '124') receive partial credit.

    Falls back to exact-match if rank-bm25 is not installed.
    """
    hyp = _normalize(hypothesis)
    ref = _normalize(reference)

    if not ref:
        return 0.0
    if not hyp:
        return 0.0

    if not _BM25_AVAILABLE:
        return 1.0 if hyp == ref else 0.0

    # Character-level tokenisation
    ref_tokens = list(ref)
    hyp_tokens = list(hyp)

    bm25 = BM25Okapi([ref_tokens])
    score = bm25.get_scores(hyp_tokens)[0]

    # Normalise by the self-similarity of the reference (upper bound)
    max_score = BM25Okapi([ref_tokens]).get_scores(ref_tokens)[0]

    if max_score <= 0:
        # Degenerate corpus (empty or single-char that IDF=0)
        return 1.0 if hyp == ref else 0.0

    return float(min(1.0, max(0.0, score / max_score)))


# ---------------------------------------------------------------------------
# Reward manager
# ---------------------------------------------------------------------------

@register("bm25")
class BM25RewardManager:
    """verl-tool reward manager that grades answers with BM25 similarity."""

    def __init__(
        self,
        tokenizer,
        num_examine: int = 1,
        compute_score=None,   # unused; kept for API compatibility
        reward_fn_key: str = "data_source",
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key

        if not _BM25_AVAILABLE:
            print(
                "[BM25RewardManager] WARNING: rank-bm25 not installed. "
                "Falling back to exact-match reward. Run: pip install rank-bm25"
            )

    # ------------------------------------------------------------------

    def __call__(self, data):
        # If scores were pre-computed upstream, skip recomputation.
        if "rm_scores" in data.batch:
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros(
            data.batch["responses"].shape, dtype=torch.float32
        )

        # Per-item metrics accumulated for reward_extra_info
        scores = []
        has_answers = []
        response_lengths = []
        already_printed = 0

        for i in range(len(data)):
            data_item = data[i]
            response_ids = data_item.batch["responses"]

            # ---- valid response length --------------------------------
            if "response_length" in data_item.batch:
                valid_len = int(data_item.batch["response_length"].item())
            else:
                pad_id = (
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id
                )
                valid_len = int((response_ids != pad_id).sum().item())
            valid_len = max(1, valid_len)

            # ---- decode response -------------------------------------
            response_str = self.tokenizer.decode(
                response_ids[:valid_len], skip_special_tokens=True
            )

            # ---- ground truth ----------------------------------------
            reward_model = data_item.non_tensor_batch["reward_model"]
            if isinstance(reward_model, (str, bytes)):
                reward_model = json.loads(reward_model)
            ground_truth = str(reward_model["ground_truth"])

            # ---- score -----------------------------------------------
            has_answer = bool(re.search(r"<answer>.*?</answer>", response_str, re.DOTALL))
            model_answer = _extract_answer(response_str)
            score = _bm25_score(model_answer, ground_truth)

            reward_tensor[i, valid_len - 1] = score

            scores.append(score)
            has_answers.append(float(has_answer))
            response_lengths.append(valid_len)

            # ---- debug printing --------------------------------------
            if already_printed < self.num_examine:
                print(
                    f"\n{'='*60}\n"
                    f"[BM25 Reward] item {i}\n"
                    f"  response (tail): ...{response_str[-300:]!r}\n"
                    f"  model answer   : {model_answer!r}\n"
                    f"  ground truth   : {ground_truth!r}\n"
                    f"  has_answer tag : {has_answer}\n"
                    f"  BM25 score     : {score:.4f}\n"
                    f"{'='*60}\n"
                )
                already_printed += 1

        # ---- aggregate extra metrics ---------------------------------
        scores_t = torch.tensor(scores)
        lengths_t = torch.tensor(response_lengths, dtype=torch.float32)
        correct_mask = scores_t > 0.99   # near-exact match

        reward_extra_info = {
            "bm25_score":              scores_t.mean().item(),
            "exact_match_rate":        correct_mask.float().mean().item(),
            "has_answer_rate":         sum(has_answers) / len(has_answers),
            "response_length":         lengths_t.mean().item(),
            "correct_response_length": lengths_t[correct_mask].mean().item() if correct_mask.any() else 0.0,
            "wrong_response_length":   lengths_t[~correct_mask].mean().item() if (~correct_mask).any() else 0.0,
        }

        return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
