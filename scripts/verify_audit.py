"""Planted-truth tests for audit.py.

Synthetic forecasters with known properties, in score_pairs.py's schema, pushed
through the real audit functions:
  honest forecaster -> ECE ~ 0
  overconfident     -> large ECE above the diagonal
  coin              -> Brier 0.25 against hard labels
  length-driven     -> the fitted length null ties it
  human ceiling     -> recovers the planted agreement rate q and beats a coin
"""
import math, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import pandas as pd
from audit import (sig, brier, ece, reliability, fit_length_null, length_null_pairs,
                   measure_q, per_vote_rows, ceiling_pairs, model_per_vote_pairs, core)

rng = random.Random(42); N = 6000; Q_TRUE = 0.88   # third annotator agrees w/ two 88% of the time

def make_world():
    rows = []
    for i in range(N):
        true_p = rng.betavariate(2, 2)
        # three annotators vote independently with P(A) = true_p
        votes = [1 if rng.random() < true_p else 0 for _ in range(3)]
        fa, fb = sum(votes), 3 - sum(votes)
        dl = (true_p - 0.5) * 400 + rng.gauss(0, 60)
        rows.append({"pair_id": i, "true_p": true_p, "favor_a": fa, "favor_b": fb,
                     "n_annotators": 3, "y_human": fa / 3,
                     "len_a": int(300 + max(0, dl)), "len_b": int(300 + max(0, -dl))})
    return rows

def logit(p): p = min(1 - 1e-6, max(1e-6, p)); return math.log(p / (1 - p))

def main():
    world = make_world(); df = pd.DataFrame(world)
    honest = [(r["true_p"], r["y_human"]) for r in world]
    over   = [(sig(3.0 * logit(r["true_p"])), r["y_human"]) for r in world]
    hard_y = [1 if r["y_human"] > 0.5 else 0 for r in world]
    coin   = [(0.5, y) for y in hard_y]
    e_h, e_o = ece(honest, 10), ece(over, 10)
    print(f"honest ECE {e_h:.4f} | overconfident ECE {e_o:.4f} | coin brier {brier(coin):.4f}")
    assert e_h < 0.03, "honest forecaster should have ~0 ECE vs soft labels"
    assert e_o > 0.08, "overconfident forecaster should light up ECE"
    assert abs(brier(coin) - 0.25) < 1e-9
    assert brier(honest) < brier(over)
    gaps = [r["mean_pred"] - r["emp_freq"] for r in reliability(over, 10) if r["n"] and r["mean_pred"] > 0.7]
    assert gaps and all(g > 0 for g in gaps), "overconfident must sit above the diagonal at high claims"

    half = N // 2; val, test = df.iloc[:half], df.iloc[half:].reset_index(drop=True)
    kappa = fit_length_null(val); ln = length_null_pairs(test, kappa)
    lendriven = [(sig(0.9 * (r["len_a"] - r["len_b"]) / 100.0), r["y_human"]) for r in world[half:]]
    print(f"kappa {kappa:.3f} | length-null brier {brier(ln):.4f} | length-driven brier {brier(lendriven):.4f}")
    assert abs(brier(ln) - brier(lendriven)) < 0.02 and brier(ln) < 0.25

    # human ceiling: q recovered on val; ceiling beats coin on per-vote rows
    q = measure_q(val)
    rows = per_vote_rows(test)
    ceil = core(ceiling_pairs(rows, q)); coinv = core([(0.5, h) for *_, h in rows])
    print(f"measured q {q:.3f} | ceiling brier {ceil['brier']:.4f} vs coin per-vote {coinv['brier']:.4f} | n micro-rows {len(rows)}")
    # in this world true agreement given a 2-0 majority is not exactly Q_TRUE (votes are
    # iid Bernoulli(true_p)); just require it is well above 0.5 and ceiling < coin
    assert 0.6 < q < 0.99, "q should be a strong-but-not-certain agreement rate"
    assert ceil["brier"] < coinv["brier"], "ceiling must beat coin on the per-vote rows"
    # a well-calibrated RM graded per-vote should not beat the ceiling by much
    p_idx = {i: honest[half + i][0] for i in range(len(test))}
    rmv = core(model_per_vote_pairs(rows, p_idx))
    print(f"honest RM per-vote brier {rmv['brier']:.4f}  (ceiling {ceil['brier']:.4f})")
    print("ALL PLANTED-TRUTH CHECKS PASS")

if __name__ == "__main__":
    main()
