"""Train one Bradley-Terry reward model with TRL's RewardTrainer.

One seed, one train fraction, one epoch. max_length is None on purpose: in TRL
it is a filter, not a truncation, and 1024 would drop ~40% of pairs unevenly
by domain. Batch size 1 with gradient accumulation keeps the longest sequence
in memory.

Usage:  python scripts/train_rm.py --seed 0 --frac 1.0
        python scripts/train_rm.py --smoke        (tiny model, 4 steps, CPU)
"""
import argparse, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of train pairs to use")
    ap.add_argument("--model", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny model, few steps, CPU-safe")
    args = ap.parse_args()

    from datasets import load_from_disk
    from trl import RewardConfig, RewardTrainer

    model_name = args.model or (CFG["smoke_model"] if args.smoke else CFG["model_name"])
    run_name = f"s{args.seed}_f{args.frac}" + ("_smoke" if args.smoke else "")
    out_dir = ROOT / "runs" / run_name

    train = load_from_disk(str(ROOT / "data" / "train"))
    # The trainer reads only `chosen` / `rejected`; the eval-side columns
    # (annotator counts, y_human, domain, strength) must not reach training.
    train = train.select_columns(["chosen", "rejected"])
    if args.frac < 1.0:
        keep = round(len(train) * args.frac)
        idx = list(range(len(train)))
        random.Random(args.seed).shuffle(idx)          # seeded: rerunnable
        train = train.select(sorted(idx[:keep]))
    if args.smoke:
        train = train.select(range(min(32, len(train))))

    cfg = RewardConfig(
        output_dir=str(out_dir),
        seed=args.seed,
        num_train_epochs=CFG["epochs"],
        learning_rate=CFG["lr"],
        per_device_train_batch_size=2 if args.smoke else CFG["per_device_bs"],
        gradient_accumulation_steps=1 if args.smoke else CFG["grad_accum"],
        max_length=256 if args.smoke else CFG["max_length"],  # None => no filter
        bf16=not args.smoke,
        gradient_checkpointing=not args.smoke,
        # Bradley-Terry loss only constrains the score difference; this term
        # pulls the mean toward zero without touching the gap.
        center_rewards_coefficient=CFG["center_rewards_coefficient"],
        logging_steps=5 if args.smoke else 50,
        save_strategy="no",
        report_to="none",
        max_steps=4 if args.smoke else -1,
    )
    trainer = RewardTrainer(model=model_name, args=cfg, train_dataset=train)
    trainer.train()
    trainer.save_model(str(out_dir))
    if trainer.processing_class is not None:
        trainer.processing_class.save_pretrained(str(out_dir))
    (out_dir / "run_meta.json").write_text(json.dumps(
        {"model": model_name, "seed": args.seed, "frac": args.frac,
         "n_train": len(train), "smoke": args.smoke}, indent=2))
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
