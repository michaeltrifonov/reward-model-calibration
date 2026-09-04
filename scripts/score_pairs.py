"""Score held-out pairs with a trained reward model.

- Each pair gets a seeded coin flip deciding which response is called A, so
  every model and baseline is graded against the same A/B assignment.
- y_human = favor_A / (favor_A + favor_B): 0 or 1 on unanimous pairs, 1/3 or
  2/3 on split ones. y_hard (accuracy only) is empty on split pairs.
- The two-step render (template -> string -> tokenize) is asserted to match
  the one-step path TRL trains with; a one-token mismatch would shift every
  margin in a length-correlated way.

Usage:  python scripts/score_pairs.py --run runs/s0_f1.0 --split test
"""
import argparse, csv, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/<name> directory")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="score only first N pairs (smoke)")
    args = ap.parse_args()

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    run_dir = ROOT / Path(args.run)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(str(run_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(run_dir), dtype=dtype).to(device).eval()
    # without a pad id the classification head cannot locate the last real token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id or tok.eos_token_id

    ds = load_from_disk(str(ROOT / "data" / args.split))
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    def render(ctx, response):
        return tok.apply_chat_template(
            list(ctx) + [{"role": "assistant", "content": response}], tokenize=False)

    def assert_render_parity(n=25):
        checked = 0
        for i in range(min(n, len(ds))):
            r = ds[i]
            for resp in (r["response1"], r["response2"]):
                msgs = list(r["context"]) + [{"role": "assistant", "content": resp}]
                two = tok(tok.apply_chat_template(msgs, tokenize=False),
                          add_special_tokens=False)["input_ids"]
                one = tok.apply_chat_template(msgs, tokenize=True, return_dict=False)
                if one and isinstance(one[0], (list, tuple)):
                    one = one[0]          # some versions return a batch of one
                assert list(two) == list(one), (
                    f"RENDER MISMATCH at row {i}: {len(two)} vs {len(one)} tokens. "
                    "Scoring-time text differs from training-time text. Fix before scoring.")
                checked += 1
        # the assertion must have compared something
        assert checked >= 2 * min(n, len(ds)), f"render check was vacuous ({checked} comparisons)"
        print(f"render-parity check: PASS ({checked} comparisons)")

    assert_render_parity()

    flip_rng = random.Random(CFG["split_seed"] + (0 if args.split == "test" else 1))
    texts, rows = [], []
    for i in range(len(ds)):
        r = ds[i]
        flip = flip_rng.random() < 0.5
        resp_a, resp_b = ((r["response2"], r["response1"]) if flip
                          else (r["response1"], r["response2"]))
        fav_a, fav_b = ((r["favor2"], r["favor1"]) if flip else (r["favor1"], r["favor2"]))
        n_ann = fav_a + fav_b
        y_human = (fav_a / n_ann) if n_ann else 0.5
        texts.append((render(r["context"], resp_a), render(r["context"], resp_b)))
        rows.append({
            "pair_id": i, "flipped": int(flip),
            "y_human": round(y_human, 4),
            "y_hard": "" if y_human == 0.5 else int(y_human > 0.5),
            "favor_a": fav_a, "favor_b": fav_b, "n_annotators": n_ann,
            "unanimous": int(bool(r["unanimous"])), "strength": r["strength"],
            "domain": r["domain"], "language": r["language"],
        })

    @torch.no_grad()
    def score_batch(strs):
        enc = tok(strs, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        return model(**enc).logits.squeeze(-1).float().cpu().tolist()

    out_path = ROOT / "out" / f"scores_{args.split}_{run_dir.name}.csv"
    out_path.parent.mkdir(exist_ok=True)
    fields = ["pair_id", "flipped", "y_human", "y_hard", "favor_a", "favor_b",
              "n_annotators", "unanimous", "strength", "domain", "language",
              "r_a", "r_b", "margin", "len_a", "len_b", "seed", "frac"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for start in range(0, len(rows), args.batch):
            chunk = rows[start:start + args.batch]
            t_a = [texts[start + j][0] for j in range(len(chunk))]
            t_b = [texts[start + j][1] for j in range(len(chunk))]
            ra, rb = score_batch(t_a), score_batch(t_b)
            for j, row in enumerate(chunk):
                row.update({
                    "r_a": round(ra[j], 6), "r_b": round(rb[j], 6),
                    "margin": round(ra[j] - rb[j], 6),
                    "len_a": len(tok(t_a[j], add_special_tokens=False)["input_ids"]),
                    "len_b": len(tok(t_b[j], add_special_tokens=False)["input_ids"]),
                    "seed": meta["seed"], "frac": meta["frac"],
                })
                w.writerow(row)
            if (start // args.batch) % 20 == 0:
                print(f"{start + len(chunk)}/{len(rows)}", flush=True)
    print(f"wrote {out_path}")
    # Raw margins are stored, never probabilities: sigma (and its temperature)
    # is applied at audit time so the band re-scores without re-inference.


if __name__ == "__main__":
    main()
