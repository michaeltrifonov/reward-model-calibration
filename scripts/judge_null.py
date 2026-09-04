"""Zero-shot judge baseline: an instruct model shown both responses, asked for
A or B, with p(A) read from the softmax over the A/B tokens.

Writes out/judge_test.csv (pair_id, p, y_human, ab_mass) on the same seeded
A/B assignment as score_pairs.py.
"""
import argparse, csv, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config.json").read_text())

PROMPT = """Two AI assistants answered the same user. Which answer would a typical human rater prefer?

[Conversation A]
{a}

[Conversation B]
{b}

Reply with exactly one letter: A or B."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--model", default=CFG["judge_null_model"])
    args = ap.parse_args()

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device).eval()
    # Both bare and space-prefixed variants; the model may emit either.
    ids_a = [tok(t, add_special_tokens=False)["input_ids"][0] for t in ("A", " A")]
    ids_b = [tok(t, add_special_tokens=False)["input_ids"][0] for t in ("B", " B")]

    ds = load_from_disk(str(ROOT / "data" / "test"))
    # same seeded flips as score_pairs.py -> same A/B, same y
    flip_rng = random.Random(CFG["split_seed"])
    flips = [flip_rng.random() < 0.5 for _ in range(len(ds))]
    sub = random.Random(99).sample(range(len(ds)), min(args.n, len(ds)))

    def flat(msgs):
        return "\n".join(f"{m['role']}: {m['content']}" for m in msgs)[:6000]

    out = ROOT / "out" / "judge_test.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "p", "y_human", "ab_mass"])
        w.writeheader()
        for i in sub:
            r = ds[i]
            a, b = (r["response2"], r["response1"]) if flips[i] else (r["response1"], r["response2"])
            fa, fb = (r["favor2"], r["favor1"]) if flips[i] else (r["favor1"], r["favor2"])
            y_h = fa / (fa + fb) if (fa + fb) else 0.5
            a = list(r["context"]) + [{"role": "assistant", "content": a}]
            b = list(r["context"]) + [{"role": "assistant", "content": b}]
            msgs = [{"role": "user", "content": PROMPT.format(a=flat(a), b=flat(b))}]
            # enable_thinking=False: otherwise Qwen3's next token is "<think>" and
            # the A-vs-B logits there are noise.
            enc = tok.apply_chat_template(msgs, return_tensors="pt", return_dict=True,
                                          add_generation_prompt=True,
                                          enable_thinking=False).to(device)
            with torch.no_grad():
                logits = model(**enc).logits[0, -1].float()
            probs = torch.softmax(logits, dim=-1)
            pa = sum(probs[t].item() for t in ids_a); pb = sum(probs[t].item() for t in ids_b)
            p_a = pa / (pa + pb) if (pa + pb) > 0 else 0.5
            w.writerow({"pair_id": i, "p": round(p_a, 6), "y_human": round(y_h, 4),
                        "ab_mass": round(pa + pb, 4)})
    print(f"wrote {out} ({len(sub)} pairs)")


if __name__ == "__main__":
    main()
