"""Audit battery. Reads out/scores_test_<run>.csv, writes out/audit_report.json.

Treats the RM as a forecaster, p = sigmoid(margin / T), and computes:
  1. Brier, log-loss, accuracy (majority rows only), ECE across 8 binnings,
     reliability curve. Graded against y_human (soft targets).
  2. Null baselines on the same rows: coin, length null (one parameter, fit on
     val), zero-shot judge (from judge_null.py), and a human ceiling
     (leave-one-annotator-out; when the visible votes agree it predicts the
     measured agreement rate q from val, graded per hidden vote alongside the
     RM graded the same way).
  3. Band: every run x every temperature. Headline = worst Brier/ECE at T=1.
  4. Slices: length-gap quartile, domain, strength, unanimous vs split.

val is used only for the length null's parameter, q, and the temperature fit.
"""
import glob, json, math
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config.json").read_text())
EPS = 1e-6


# ---------- core metrics (soft-target aware) ---------------------------------------
def sig(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x); return e / (1.0 + e)

def brier(pairs):   return sum((p - y) ** 2 for p, y in pairs) / len(pairs)

def logloss(pairs):
    s = 0.0
    for p, y in pairs:
        pc = min(1 - EPS, max(EPS, p))
        s += -(y * math.log(pc) + (1 - y) * math.log(1 - pc))
    return s / len(pairs)

def accuracy(pairs):
    """Only rows with a majority count; a 50/50 human split has no 'right side'."""
    dec = [(p, y) for p, y in pairs if abs(y - 0.5) > 1e-9]
    if not dec: return float("nan")
    return sum(1 for p, y in dec if (p >= 0.5) == (y > 0.5)) / len(dec)

def _bins_width(pairs, nb):
    b = [[] for _ in range(nb)]
    for p, y in pairs: b[min(nb - 1, max(0, int(p * nb)))].append((p, y))
    return b

def _bins_mass(pairs, nb):
    s = sorted(pairs)
    return [s[round(i * len(s) / nb): round((i + 1) * len(s) / nb)] for i in range(nb)]

def ece(pairs, nb=10, scheme="width"):
    n, e = len(pairs), 0.0
    for sel in (_bins_width if scheme == "width" else _bins_mass)(pairs, nb):
        if sel:
            mp = sum(p for p, _ in sel) / len(sel); ef = sum(y for _, y in sel) / len(sel)
            e += abs(mp - ef) * len(sel) / n
    return e

def reliability(pairs, nb=10):
    rows = []
    for b, sel in enumerate(_bins_width(pairs, nb)):
        lo, hi = b / nb, (b + 1) / nb
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(sel),
                     "mean_pred": round(sum(p for p, _ in sel) / len(sel), 3) if sel else None,
                     "emp_freq": round(sum(y for _, y in sel) / len(sel), 3) if sel else None})
    return rows

def core(pairs):
    return {"n": len(pairs), "accuracy": round(accuracy(pairs), 4),
            "brier": round(brier(pairs), 4), "log_loss": round(logloss(pairs), 4),
            "ece_10w": round(ece(pairs, 10, "width"), 4)}

def ece_sweep(pairs):
    return {f"{sch}_{nb}": round(ece(pairs, nb, sch), 4)
            for sch in ("width", "mass") for nb in CFG["ece_bin_grid"]}


# ---------- length null (kappa fit on val only) ----------------------------------
def fit_length_null(val_df):
    dl = [(a - b) / 100.0 for a, b in zip(val_df.len_a, val_df.len_b)]
    ys = list(val_df.y_human)
    def nll(k):
        s = 0.0
        for d, y in zip(dl, ys):
            p = min(1 - EPS, max(EPS, sig(k * d)))
            s += -(y * math.log(p) + (1 - y) * math.log(1 - p))
        return s
    k = min([i / 50.0 for i in range(-250, 251)], key=nll)
    for step in (0.01, 0.002):
        k = min([k + j * step for j in range(-10, 11)], key=nll)
    return k

def length_null_pairs(df, kappa):
    return [(sig(kappa * (a - b) / 100.0), y) for a, b, y in zip(df.len_a, df.len_b, df.y_human)]


# ---------- human ceiling (leave-one-annotator-out) ------------------------------
def measure_q(val_df):
    """On VAL: when the other annotators have a majority, how often does the
    held-out annotator agree with it? This is the ceiling's confidence."""
    agree = total = 0
    for fa, fb in zip(val_df.favor_a, val_df.favor_b):
        fa, fb = int(fa), int(fb)
        if fa + fb < 3: continue
        for held_a in (True, False):                 # hold out an A-voter / a B-voter
            k = fa if held_a else fb                 # how many such holdouts exist
            ra, rb = (fa - 1, fb) if held_a else (fa, fb - 1)
            if ra == rb or k == 0: continue          # no majority to agree with
            majority_a = ra > rb
            agree += k * (1 if majority_a == held_a else 0)
            total += k
    return agree / total if total else 0.9

def per_vote_rows(df, min_annot=3):
    """Expand each >=3-annotator pair into one micro-row per held-out annotator:
    (pair index, remaining_a, remaining_b, hidden_vote). The RM and the ceiling
    are both graded on these SAME micro-rows, against the hidden vote."""
    out = []
    for i, (fa, fb) in enumerate(zip(df.favor_a, df.favor_b)):
        fa, fb = int(fa), int(fb)
        if fa + fb < min_annot: continue
        for _ in range(fa): out.append((i, fa - 1, fb, 1))
        for _ in range(fb): out.append((i, fa, fb - 1, 0))
    return out

def ceiling_pairs(rows, q):
    ps = []
    for _, ra, rb, hidden in rows:
        p = 0.5 if ra == rb else (q if ra > rb else 1 - q)
        ps.append((p, hidden))
    return ps

def model_per_vote_pairs(rows, p_by_index):
    return [(p_by_index[i], hidden) for i, _, _, hidden in rows]


# ---------- slices ---------------------------------------------------------------
def slice_table(df, pairs, key_fn, edges, label):
    rows = []
    for lo, hi, name in edges:
        sel = [pairs[i] for i in range(len(pairs)) if lo <= key_fn(df.iloc[i]) < hi]
        if len(sel) >= 30:
            rows.append({**core(sel), "slice": name})
    return {"slice_by": label, "rows": rows}

def cat_slice(df, pairs, col, label):
    rows = []
    for v in sorted(df[col].unique(), key=str):
        sel = [pairs[i] for i in range(len(pairs)) if df.iloc[i][col] == v]
        if len(sel) >= 30:
            rows.append({**core(sel), "slice": str(v)})
    return {"slice_by": label, "rows": rows}

def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]: j += 1
            for t in range(i, j + 1): r[order[t]] = (i + j) / 2 + 1
            i = j + 1
        return r
    if len(xs) < 3: return float("nan")
    rx, ry = rank(xs), rank(ys); mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


# ---------- main -----------------------------------------------------------------
def main():
    files = sorted(f for f in glob.glob(str(ROOT / "out" / "scores_test_*.csv")) if "smoke" not in f)
    if not files:
        raise SystemExit("no out/scores_test_*.csv — run score_pairs.py first")
    runs = {Path(f).stem.replace("scores_test_", ""): pd.read_csv(f) for f in files}
    primary_name = "s0_f1.0" if "s0_f1.0" in runs else sorted(runs)[0]; primary = runs[primary_name]
    R = {"primary_run": primary_name, "runs": sorted(runs)}
    to_pairs = lambda df, T=1.0: [(sig(m / T), y) for m, y in zip(df.margin, df.y_human)]

    # 1. core metrics
    pairs = to_pairs(primary)
    R["core"] = core(pairs); R["reliability"] = reliability(pairs); R["ece_sensitivity"] = ece_sweep(pairs)

    # 2. null baselines
    R["null_coin"] = core([(0.5, y) for _, y in pairs])
    val_file = ROOT / "out" / f"scores_val_{primary_name}.csv"
    if val_file.exists():
        val = pd.read_csv(val_file)
        kappa = fit_length_null(val); ln = length_null_pairs(primary, kappa)
        R["null_length"] = {**core(ln), "kappa": round(kappa, 4)}
        R["reliability_length_null"] = reliability(ln); R["ece_sensitivity_length_null"] = ece_sweep(ln)
        R["rm_beats_length_null"] = {k: (R["core"][k] < R["null_length"][k]) if k != "accuracy"
                                     else (R["core"][k] > R["null_length"][k])
                                     for k in ("accuracy", "brier", "ece_10w")}
        # human ceiling: per-vote table, RM graded the same way on the same micro-rows
        q = measure_q(val)
        rows = per_vote_rows(primary)
        p_idx = {i: p for i, (p, _) in enumerate(pairs)}
        R["human_ceiling"] = {"q_measured_on_val": round(q, 4), "n_micro_rows": len(rows),
                              "ceiling": core(ceiling_pairs(rows, q)),
                              "rm_per_vote": core(model_per_vote_pairs(rows, p_idx)),
                              "coin_per_vote": core([(0.5, h) for *_, h in rows]),
                              "note": "graded against each hidden annotator's vote; "
                                      "compare rows within this table only"}
    else:
        print("NOTE: no val scores for primary run -> length null + ceiling skipped")
    jf = ROOT / "out" / "judge_test.csv"
    if jf.exists():
        jd = pd.read_csv(jf); jp = list(zip(jd.p, jd.y_human))
        R["null_judge"] = {**core(jp), "mean_p": round(float(jd.p.mean()), 4),
                           "note": "mean_p far from 0.5 = position bias"}
        R["reliability_judge"] = reliability(jp); R["ece_sensitivity_judge"] = ece_sweep(jp)

    # 3. band
    band = [{"run": n, "T": T, **core(to_pairs(df, T))} for n, df in runs.items() for T in CFG["temperatures"]]
    R["band"] = band
    # headline = worst across runs at native T=1; temperature is reported separately
    native = [b for b in band if b["T"] == 1.0]
    R["headline"] = {"worst_brier": max(b["brier"] for b in native), "worst_ece_10w": max(b["ece_10w"] for b in native),
                     "best_brier": min(b["brier"] for b in native), "n_runs": len(native),
                     "note": "worst across seeds x train-frac at native T=1; temperature reported separately"}
    R["temperature_sensitivity"] = {f"T={T}": {"worst_ece": max(b["ece_10w"] for b in band if b["T"] == T),
                                               "worst_brier": max(b["brier"] for b in band if b["T"] == T)}
                                    for T in CFG["temperatures"]}

    # 4. slices
    dl_abs = [abs(a - b) for a, b in zip(primary.len_a, primary.len_b)]
    q_ = sorted(dl_abs); q1, q2, q3 = q_[len(q_) // 4], q_[len(q_) // 2], q_[3 * len(q_) // 4]
    R["slice_length_gap"] = slice_table(primary, pairs, lambda r: abs(r.len_a - r.len_b),
        [(0, q1, f"|dlen|<{q1}"), (q1, q2, f"{q1}-{q2}"), (q2, q3, f"{q2}-{q3}"), (q3, 1e9, f">={q3}")], "abs token-length gap")
    R["slice_domain"] = cat_slice(primary, pairs, "domain", "domain")
    R["slice_strength"] = cat_slice(primary, pairs, "strength", "human preference strength (0 = split)")
    R["slice_agreement"] = cat_slice(primary, pairs, "unanimous", "annotators unanimous (1) vs split (0)")
    conf = [abs(p - 0.5) for p, _ in pairs]
    dec = [i for i, (_, y) in enumerate(pairs) if abs(y - 0.5) > 1e-9]
    correct = {i: (pairs[i][0] >= 0.5) == (pairs[i][1] > 0.5) for i in dec}
    R["verbosity_correlation"] = {
        "spearman_conf_vs_lengap_all": round(spearman(conf, dl_abs), 4),
        "spearman_conf_vs_lengap_correct_only": round(spearman([conf[i] for i in dec if correct[i]], [dl_abs[i] for i in dec if correct[i]]), 4),
        "spearman_conf_vs_lengap_wrong_only": round(spearman([conf[i] for i in dec if not correct[i]], [dl_abs[i] for i in dec if not correct[i]]), 4),
        "note": "spearman between |p-0.5| and |len_a-len_b|, split by correctness"}

    out = ROOT / "out" / "audit_report.json"; out.write_text(json.dumps(R, indent=2))

    # ---- print ----
    c = R["core"]; print("=" * 78)
    print(f"AUDIT — primary {primary_name} (T=1), n={c['n']}  |  acc {c['accuracy']:.3f}  brier {c['brier']:.4f}  ll {c['log_loss']:.4f}  ece {c['ece_10w']:.4f}")
    print("=" * 78); print(f"{'bin':<10}{'n':>6}{'mean_pred':>11}{'emp_freq':>10}{'gap':>8}")
    for r in R["reliability"]:
        if r["n"]: print(f"{r['bin']:<10}{r['n']:>6}{r['mean_pred']:>11.3f}{r['emp_freq']:>10.3f}{r['mean_pred'] - r['emp_freq']:>8.3f}")
    print("-" * 78); print("baselines (graded vs y_human):")
    print(f"  {'coin flip':<14} brier {R['null_coin']['brier']:.4f}")
    if "null_length" in R:
        n = R["null_length"]; print(f"  {'length null':<14} brier {n['brier']:.4f}  ece {n['ece_10w']:.4f}  acc {n['accuracy']:.3f}  (kappa {n['kappa']})")
    if "null_judge" in R:
        n = R["null_judge"]; print(f"  {'zero-shot judge':<14} brier {n['brier']:.4f}  ece {n['ece_10w']:.4f}  acc {n['accuracy']:.3f}")
    print(f"  {'trained RM':<14} brier {c['brier']:.4f}  ece {c['ece_10w']:.4f}  acc {c['accuracy']:.3f}")
    if "human_ceiling" in R:
        h = R["human_ceiling"]; print("-" * 78)
        print(f"PER-VOTE TABLE (n={h['n_micro_rows']} hidden votes, q={h['q_measured_on_val']}):")
        for k in ("coin_per_vote", "rm_per_vote", "ceiling"):
            print(f"  {k:<14} brier {h[k]['brier']:.4f}  ece {h[k]['ece_10w']:.4f}  acc {h[k]['accuracy']:.3f}")
    print("-" * 78)
    print("BAND (every run x T; headline = worst):")
    for b in band:
        print(f"  {b['run']:<10} T={b['T']:<4} acc {b['accuracy']:.3f}  brier {b['brier']:.4f}  ece {b['ece_10w']:.4f}")
    hd = R["headline"]; print(f"HEADLINE (worst of {hd['n_runs']} runs, native T=1): brier {hd['worst_brier']:.4f}  ece {hd['worst_ece_10w']:.4f}   (best brier {hd['best_brier']:.4f})")
    print("temperature sensitivity (reported, not headlined):", {k: v["worst_ece"] for k, v in R["temperature_sensitivity"].items()})
    print(f"verbosity: {R['verbosity_correlation']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
