"""Reward functions for differential-level GRPO on DDXPlus.

Every function takes the completion text plus the case's gold condition and ground-truth
differential and returns a float. The trainer wrappers at the bottom adapt them to TRL's
reward-function signature (prompts, completions, **dataset_columns).

    format    1 if the answer parses into an on-list differential with valid probabilities

Names are canonicalized with the same matcher used for evaluation (ddx_names.Matcher), so
"Acute bronchitis" or "Pulmonary Embolism" count as the listed condition; `allowed` may be a
Matcher or a plain set of exact names.
    acc       1 if the top-1 diagnosis is the gold condition (standard RLVR reward)
    brier     1 - Brier score of the stated distribution against the gold outcome
              (a strictly proper scoring rule: maximized by reporting true beliefs)
    logscore  log-probability assigned to the gold, clipped at log(eps) and rescaled to [0, 1]
    overlap   sum_c min(p_model(c), p_gt(c)) = 1 - total variation distance between the stated
              differential and the ground-truth differential (rewards covering the differential)
"""
import json
import math

from ddx_names import Matcher
from parsing import parse_dx

EPS = 1e-3


def _canon(allowed):
    return allowed if callable(allowed) else (lambda n: n if n in allowed else None)


def canonical_dx(text, allowed):
    """[(canonical name or None, prob)] in listed order, or None if nothing parses."""
    dx = parse_dx(text)
    if not dx:
        return None
    canon = _canon(allowed)
    return [(canon(d["diagnosis"]), d["probability"]) for d in dx]


def parse_distribution(text, allowed):
    """Return (dist, failure_reason). dist is {condition: prob} or None.

    Probabilities are renormalized when they sum above 1 (models often list independent
    likelihoods); leftover mass below 1 is treated as 'other' and earns nothing.
    failure_reason is one of: unparsed, offlist, duplicate, missing_prob, bad_prob, zero_total.
    """
    dx = canonical_dx(text, allowed)
    if not dx:
        return None, "unparsed"
    dist = {}
    for name, p in dx:
        if name is None:
            return None, "offlist"
        if name in dist:
            return None, "duplicate"
        if p is None:
            return None, "missing_prob"
        if not 0 <= p <= 1:
            return None, "bad_prob"
        dist[name] = p
    total = sum(dist.values())
    if total <= 0:
        return None, "zero_total"
    if total > 1:
        dist = {k: v / total for k, v in dist.items()}
    return dist, None


def stated_distribution(text, allowed):
    return parse_distribution(text, allowed)[0]


def top1(text, allowed):
    dx = canonical_dx(text, allowed)
    return dx[0][0] if dx else None


def r_format(text, allowed, **_):
    return 1.0 if stated_distribution(text, allowed) is not None else 0.0


def r_acc(text, gold, allowed, **_):
    return 1.0 if top1(text, allowed) == gold else 0.0


def r_brier(text, gold, allowed, **_):
    dist = stated_distribution(text, allowed)
    if dist is None:
        return 0.0
    # multiclass Brier over listed conditions + implicit 'other' bucket; range [0, 2]
    other = max(0.0, 1.0 - sum(dist.values()))
    bs = sum((p - (1.0 if c == gold else 0.0)) ** 2 for c, p in dist.items())
    bs += other ** 2 + (0.0 if gold in dist else 1.0)
    return 1.0 - bs / 2.0


def r_logscore(text, gold, allowed, **_):
    dist = stated_distribution(text, allowed)
    if dist is None:
        return 0.0
    p = max(dist.get(gold, 0.0), EPS)
    return 1.0 - math.log(p) / math.log(EPS)  # 0 at p=eps, 1 at p=1


def r_overlap(text, gt_differential, allowed, **_):
    dist = stated_distribution(text, allowed)
    if dist is None:
        return 0.0
    gt = dict(gt_differential)
    z = sum(gt.values())
    return sum(min(p, gt.get(c, 0.0) / z) for c, p in dist.items())


# --- Baselines from Puri et al. 2026, "Reaching Beyond the Mode" (arXiv 2603.24844) ---
# They treat the differential as a SET (Y*) and reward set membership of K distinct answers;
# Multi-Answer RLCR subtracts a set-level Brier term on per-answer confidences. Here Y* is the
# set of conditions in the DDXPlus ground-truth differential, K = number of listed answers.

def _listed(text, allowed):
    dx = canonical_dx(text, allowed)
    if not dx:
        return None
    names = [n for n, _ in dx]
    if len(set(names)) != len(names) or None in names:
        return None  # Puri et al. zero the reward when answers are not unique/valid
    return dx


def r_multi_rlvr(text, gt_differential, allowed, **_):
    dx = _listed(text, allowed)
    if dx is None:
        return 0.0
    gold_set = {c for c, _ in gt_differential}
    return sum(n in gold_set for n, _ in dx) / len(dx)


def r_set_brier(text, gt_differential, allowed, **_):
    """1 - mean_i (q_i - 1[a_i in Y*])^2; adding this == their RLVR_multi - Brier_multi (up to a constant)."""
    dx = _listed(text, allowed)
    if dx is None or any(p is None or not 0 <= p <= 1 for _, p in dx):
        return 0.0
    gold_set = {c for c, _ in gt_differential}
    return 1.0 - sum((p - (n in gold_set)) ** 2 for n, p in dx) / len(dx)


REWARDS = {"format": r_format, "acc": r_acc, "brier": r_brier, "logscore": r_logscore, "overlap": r_overlap,
           "multi_rlvr": r_multi_rlvr, "set_brier": r_set_brier}


# --- Group-level reward: match the distribution of SAMPLED top-1 answers to the reference ---
# Per-sample rewards above only see one completion's stated list, so they cannot see that the
# policy gives the same top-1 answer in every sample (the collapse we measure). group_match looks
# at the whole GRPO group for a case: with n samples, k_a of which answer a, sample i (answer a)
# gets its marginal contribution to n * sum_c min(q_c, p_c) (overlap of the group's empirical
# distribution q with the reference p, fixed denominator n):
#     r_i = clip(n * p_a - (k_a - 1), 0, 1)
# i.e. full credit while answer a is under-represented relative to its reference mass, none once
# it is over-represented. Invalid / off-list / zero-reference answers get 0.

def group_match_rewards(texts, case_ids, gt_differentials, allowed):
    tops = []
    for t in texts:
        dx = canonical_dx(t, allowed)
        tops.append(dx[0][0] if dx else None)
    groups = {}
    for i, cid in enumerate(case_ids):
        groups.setdefault(cid, []).append(i)
    out = [0.0] * len(texts)
    for idx in groups.values():
        n = len(idx)
        gt = dict(gt_differentials[idx[0]])
        z = sum(gt.values()) or 1.0
        counts = {}
        for i in idx:
            if tops[i] is not None:
                counts[tops[i]] = counts.get(tops[i], 0) + 1
        for i in idx:
            a = tops[i]
            if a is None:
                continue
            out[i] = min(1.0, max(0.0, n * gt.get(a, 0.0) / z - (counts[a] - 1)))
    return out, {len(v) for v in groups.values()}


def make_trl_reward(name, allowed):
    """Wrap a reward for TRL: f(prompts, completions, **columns) -> list[float]."""
    allowed = Matcher(allowed)
    if name == "group_match":
        def group_reward(prompts=None, completions=None, case_id=None, gt_differential=None, log_metric=None, **_):
            texts = [c[-1]["content"] if isinstance(c, list) else c for c in completions]
            gts = [json.loads(g) if isinstance(g, str) else g for g in gt_differential]
            out, sizes = group_match_rewards(texts, case_id, gts, allowed)
            if log_metric is not None:
                try:  # groups must be complete for the reward to mean anything
                    log_metric("group_match/min_group_size", float(min(sizes)))
                except Exception as e:
                    print(f"warn: log_metric failed ({e!r})")
            return out
        group_reward.__name__ = "r_group_match"
        return group_reward
    fn = REWARDS[name]

    def reward(prompts=None, completions=None, gold=None, gt_differential=None, log_metric=None, **_):
        out, reasons, tops = [], [], []
        for i, c in enumerate(completions):
            text = c[-1]["content"] if isinstance(c, list) else c  # conversational or plain
            gt = gt_differential[i]
            gt = json.loads(gt) if isinstance(gt, str) else gt  # stored as JSON (Arrow can't mix types)
            out.append(fn(text, gold=gold[i], gt_differential=gt, allowed=allowed))
            if name == "format":
                reasons.append(parse_distribution(text, allowed)[1])
                tops.append(top1(text, allowed))
        if name == "format" and log_metric is not None and reasons:
            # fraction of completions failing the format check, by reason; plus a collapse monitor:
            # distinct top-1 answers per distinct case in this batch (~1/num_generations when the
            # policy answers each case deterministically, -> 1/num_cases if it ignores the case)
            try:
                for r in ("unparsed", "offlist", "duplicate", "missing_prob", "bad_prob", "zero_total"):
                    log_metric(f"format_fail/{r}", sum(x == r for x in reasons) / len(reasons))
                log_metric("collapse/distinct_top1_per_case", len(set(tops)) / max(1, len(set(gold))))
            except Exception as e:  # diagnostics must never break training
                print(f"warn: log_metric failed ({e!r}); format-failure metrics not logged")
        return out

    reward.__name__ = f"r_{name}"
    return reward
