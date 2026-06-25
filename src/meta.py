"""REPRO_CONTRACT sidecars, per-figure CSVs, sources.json for rl-selfcot-causal."""
import csv, json
from pathlib import Path

OUT = Path("results/real")
GIT = "12da608880867be46ae58a5e53a78ef6e4e248d9"
rows = list(csv.DictReader(open(OUT / "curve.csv")))


def sidecar(name, columns, analysis_cmd):
    (OUT / (name + ".md")).write_text(
"""# %s — sidecar (REPRO_CONTRACT)

Generated-By: src/eval_corruption.py (checkpoints from src/train_grpo.py)
Command: python3 src/eval_corruption.py --base_adapter runs/rl/checkpoint-0 --rl_adapter runs/rl/checkpoint-final --n_eval 300 --seed 42
Git-Commit: %s
Seeds: 42 (MMLU eval-slice sampling in src/task.py; greedy decoding so generation is deterministic; 2000-resample percentile bootstrap for CIs; GRPO training seed 0)
Source-Data: MMLU (cais/mmlu) reserved eval slice via src/task.py load_items('mmlu','eval'); model Qwen2.5-1.5B-Instruct adapted by outcome-only GRPO (reward=answer-correctness, no cue) to runs/rl/checkpoint-0 (base anchor, pre-RL) and runs/rl/checkpoint-final (after 120 steps), RTX 5090, 2026-06-24, torch 2.12 cu130, trl 1.5.1
Analysis-Command: %s
Columns:
%s
""" % (name, GIT, analysis_cmd, columns))


sidecar("curve.csv",
        "  model (base = checkpoint-0 pre-RL anchor, rl = checkpoint-final after outcome-only GRPO);\n"
        "  metric (accuracy = own-CoT greedy MMLU accuracy; intact_flip = control flip rate when the model's own full CoT is prefilled; truncate_flip = first-half CoT; mismatch_flip = another item's CoT; nocot_flip = answer forced with no reasoning; mean_corruption_flip = mean of truncate+mismatch+nocot);\n"
        "  value (rate or accuracy, unitless 0-1); n (items kept: parseable self-answer and non-empty CoT)",
        "cd results/real && python3 recompute.py | diff - analysis_summary.txt  (empty); curve.csv values equal the per-section rates recompute prints")

sidecar("eval_points.jsonl",
        "  section (e.g. base_mismatch_flip, rl_accuracy); eval_order (position within section); val (0/1: for *_flip = did the final answer change under that corruption; for *_accuracy = was the own-CoT answer correct)",
        "cd results/real && python3 recompute.py  (each section rate = mean(val) with 95% bootstrap CI)")

def w(stem, rs, desc):
    cols = ["model", "metric", "value", "n"]
    with open(OUT / (stem + ".csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader()
        for r in rs:
            wr.writerow({c: r[c] for c in cols})
    (OUT / (stem + ".md")).write_text(
        "# %s.csv / %s.png\n\n%s\n\nSource: curve.csv (slice). Generated-By: src/eval_corruption.py + src/meta.py. Git-Commit: %s\n" % (stem, stem, desc, GIT))

flips = [r for r in rows if r["metric"].endswith("_flip") and r["metric"] != "mean_corruption_flip"]
w("figure_main", flips, "Flip rate by corruption condition (intact/truncate/mismatch/nocot), base vs RL.")
w("figure_by_condition", flips, "Flip rate by corruption type, base vs RL (line view).")
w("figure_accuracy", [r for r in rows if r["metric"] == "accuracy"], "Own-CoT MMLU accuracy, base vs RL.")
w("figure_causal_dependence", [r for r in rows if r["metric"] == "mean_corruption_flip"], "Mean corruption flip rate (causal dependence on CoT), base vs RL.")

(OUT / "sources.json").write_text(json.dumps({"metrics": {"*": {"csv": "curve.csv"}}, "per_example": ["eval_points.jsonl"]}, indent=2))
print("wrote sidecars + per-figure csv/md + sources.json")
