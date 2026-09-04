"""Temperature scaling, fit on val and reported only.

Fits T so sigmoid(margin / T) minimises log-loss on val, then reports what it
would do to the test metrics. It is never applied to the headline.

Usage: python scripts/temp_scale.py --run-name s0_f1.0
"""
import argparse, json, math
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
import sys; sys.path.insert(0, str(ROOT / "scripts"))
from audit import sig, core, EPS  # same instruments, no duplicates


def nll(df, T):
    s = 0.0
    for m, y in zip(df.margin, df.y_human):
        p = min(1 - EPS, max(EPS, sig(m / T)))
        s += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return s / len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="s0_f1.0")
    args = ap.parse_args()
    val = pd.read_csv(ROOT / "out" / f"scores_val_{args.run_name}.csv")
    test = pd.read_csv(ROOT / "out" / f"scores_test_{args.run_name}.csv")

    # golden-section search on log T in [1/8, 8]
    lo, hi = math.log(0.125), math.log(8.0)
    phi = (math.sqrt(5) - 1) / 2
    a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fa, fb = nll(val, math.exp(a)), nll(val, math.exp(b))
    for _ in range(40):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - phi * (hi - lo); fa = nll(val, math.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + phi * (hi - lo); fb = nll(val, math.exp(b))
    T = math.exp((lo + hi) / 2)

    before = core([(sig(m), y) for m, y in zip(test.margin, test.y_human)])
    after = core([(sig(m / T), y) for m, y in zip(test.margin, test.y_human)])
    out = {"fitted_T_on_val": round(T, 4),
           "test_before": before, "test_after_wouldbe": after,
           "note": "fitted on val; reported only, never applied"}
    (ROOT / "out" / f"temp_scale_{args.run_name}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("T>1 means the raw margins were overconfident (needed softening).")


if __name__ == "__main__":
    main()
