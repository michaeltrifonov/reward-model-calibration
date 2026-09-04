# reward-model-calibration

Train a small Bradley-Terry reward model on HelpSteer3-Preference and audit it as a forecaster: Brier, log-loss, ECE across binnings, reliability curves, null baselines on the same rows, slices by annotator agreement, and a temperature fit that is reported but not applied.

Model: `Qwen/Qwen3-1.7B-Base`, scalar head, full fine-tune, one epoch. Four runs: seeds 0, 1, 2 on full data and seed 0 on half. Checkpoints are not included; the per-pair scores, audit report, and figures are.

## Layout

```
config.json
setup.sh                  install pins, assert a GPU, build the splits
scripts/
  prepare_data.py         HelpSteer3 -> train/test/val
  train_rm.py             train one run
  score_pairs.py          score held-out pairs
  judge_null.py           zero-shot judge baseline
  temp_scale.py           temperature fit on val (reported only)
  audit.py                metrics, baselines, band, slices -> out/audit_report.json
  verify_audit.py         planted-truth tests for audit.py
  figures.py              figures and band table
out/
  scores_test_<run>.csv   per-pair scores for each run
  scores_val_s0_f1.0.csv
  judge_test.csv
  audit_report.json
  band_table.md
  temp_scale_s0_f1.0.json
  fig1_reliability.png  fig2_ece_binning.png  fig3_slices.png
```

## Reproduce

The audit runs without a GPU from the committed CSVs:

```bash
pip install pandas matplotlib
python scripts/verify_audit.py
python scripts/audit.py
python scripts/figures.py
```

Retraining needs one 40GB+ GPU and the pins in `requirements.txt`:

```bash
bash setup.sh
python scripts/train_rm.py --seed 0 --frac 1.0        # and seeds 1, 2; --frac 0.5
python scripts/score_pairs.py --run runs/s0_f1.0 --split test
python scripts/score_pairs.py --run runs/s0_f1.0 --split val
python scripts/judge_null.py
python scripts/temp_scale.py --run-name s0_f1.0
python scripts/audit.py && python scripts/figures.py
```

## CSV schema

`pair_id, flipped, y_human, y_hard, favor_a, favor_b, n_annotators, unanimous, strength, domain, language, r_a, r_b, margin, len_a, len_b, seed, frac`

`y_human` is the fraction of annotators who preferred A after the seeded flip. `y_hard` is empty on split pairs. `margin = r_a - r_b`; the forecast is `sigmoid(margin)`.

## License

Code is MIT. The CSVs are derived from HelpSteer3 (NVIDIA, CC-BY-4.0) and contain no response text.
