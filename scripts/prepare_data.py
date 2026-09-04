"""Build train/test/val splits from HelpSteer3-Preference.

- Byte-identical duplicate rows are merged; distinct annotator entries are pooled.
- overall_preference == 0 means the annotators disagreed on direction. Those rows
  are dropped from train (Bradley-Terry needs a winner) and kept in test/val.
- Splits are made by conversation (a context can appear in several pairs) and
  asserted disjoint.

Writes data/{train,test,val} (HF datasets on disk).
"""
import hashlib, json, random, sys
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config.json").read_text())


def context_key(ctx):
    """A conversation's identity = its full message history."""
    blob = json.dumps([[m["role"], m["content"]] for m in ctx], sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def annotator_counts(individual):
    """(favor_response1, favor_response2). Negative score = response1 better."""
    f1 = sum(1 for a in individual if a["score"] < 0)
    f2 = sum(1 for a in individual if a["score"] > 0)
    return f1, f2


def merge_duplicates(ds):
    """Collapse exact-duplicate pairs (same context, response1, response2) into
    one row, pooling their distinct annotator entries. Byte-identical copies
    collapse to one vote each; genuine re-ratings become one row with more
    annotators."""
    groups = {}
    order = []
    for i, r in enumerate(ds):
        k = hashlib.sha1((json.dumps([[m["role"], m["content"]] for m in r["context"]])
                          + r["response1"] + r["response2"]).encode("utf-8")).hexdigest()
        if k not in groups:
            groups[k] = {"row": r, "annots": {}}
            order.append(k)
        for a in r["individual_preference"]:
            ak = json.dumps([a["score"], a["reasoning"], a["feedback1"], a["feedback2"]])
            groups[k]["annots"][ak] = a          # identical entries collapse here
    merged = []
    for k in order:
        g = groups[k]; r = dict(g["row"])
        annots = list(g["annots"].values())
        scores = [a["score"] for a in annots]
        f1 = sum(1 for x in scores if x < 0); f2 = sum(1 for x in scores if x > 0)
        unanimous = (f1 == 0 or f2 == 0) and (f1 + f2) > 0
        mean = sum(scores) / len(scores) if scores else 0.0
        r["individual_preference"] = annots
        r["overall_preference"] = int(round(mean)) if unanimous else 0   # keep 0 <=> split
        merged.append(r)
    from datasets import Dataset
    return Dataset.from_list(merged)


def annotate(row):
    """Attach the eval-side fields every downstream station needs."""
    f1, f2 = annotator_counts(row["individual_preference"])
    n = f1 + f2
    return {
        "ctx_key": context_key(row["context"]),
        "n_annotators": n,
        "favor1": f1,
        "favor2": f2,
        # target: P(a random annotator prefers response1)
        "y_human": (f1 / n) if n else 0.5,
        "unanimous": (f1 == 0 or f2 == 0) and n > 0,
        "strength": abs(row["overall_preference"]),   # 0..3, 0 = humans split
    }


def to_pair_columns(row):
    """TRL RewardTrainer wants conversational `chosen` / `rejected`: the full
    context with the winning / losing answer appended as the assistant turn."""
    r1_won = row["overall_preference"] < 0
    win = row["response1"] if r1_won else row["response2"]
    lose = row["response2"] if r1_won else row["response1"]
    return {
        "chosen": list(row["context"]) + [{"role": "assistant", "content": win}],
        "rejected": list(row["context"]) + [{"role": "assistant", "content": lose}],
    }


def main():
    ds = load_dataset(CFG["dataset"], CFG["dataset_config"])
    raw_train_n, raw_val_n = len(ds["train"]), len(ds["validation"])
    pool = merge_duplicates(ds["train"]).map(annotate)
    val = merge_duplicates(ds["validation"]).map(annotate)
    print(f"dedupe-by-merge: train {raw_train_n} -> {len(pool)} distinct pairs | "
          f"val {raw_val_n} -> {len(val)}")

    # --- carve test out of their train, at CONVERSATION granularity ----------
    by_ctx = {}
    for i, k in enumerate(pool["ctx_key"]):
        by_ctx.setdefault(k, []).append(i)
    ctxs = sorted(by_ctx)
    random.Random(CFG["split_seed"]).shuffle(ctxs)

    test_idx, train_idx = [], []
    for k in ctxs:
        (test_idx if len(test_idx) < CFG["n_test"] else train_idx).extend(by_ctx[k])

    test = pool.select(test_idx)
    train_all = pool.select(train_idx)

    # --- train: unanimous rows only, then add chosen/rejected ---------------
    train = train_all.filter(lambda r: r["overall_preference"] != 0)
    train = train.map(to_pair_columns)
    if CFG["max_train"] and len(train) > CFG["max_train"]:
        train = train.select(range(CFG["max_train"]))

    out = ROOT / "data"
    for name, d in (("train", train), ("test", test), ("val", val)):
        d.save_to_disk(str(out / name))

    # --- summary -----
    def describe(name, d):
        n = len(d)
        split = sum(1 for u in d["unanimous"] if not u)
        print(f"{name:<6} {n:>6} rows | {len(set(d['ctx_key'])):>6} conversations "
              f"| humans-disagreed {split:>5} ({100*split/n:.1f}%)")
    describe("train", train); describe("test", test); describe("val", val)

    ks = {n: set(d["ctx_key"]) for n, d in (("train", train), ("test", test), ("val", val))}
    assert not (ks["test"] & ks["train"]), "LEAKAGE: shared conversation train/test"
    print("conversation-overlap check: PASS (0 shared between train and test)")
    print("NOTE: train excludes humans-disagreed rows by design "
          "(no winner to learn); test/val keep them (that's the audit's point).")


if __name__ == "__main__":
    sys.exit(main())
