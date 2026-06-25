"""recompute.py — reproduce analysis_summary.txt from per-example data alone.

Pure stdlib. Reads eval_points.jsonl ({section, eval_order, val} with val in {0,1})
and reports each section's rate = mean(val) with a seeded bootstrap 95% CI.

Gate: `cd results/real && python3 recompute.py | diff - analysis_summary.txt` empty.
"""
import json
import random
from collections import defaultdict

BOOT_N = 2000
BOOT_SEED = 42

SECTIONS = [
    ("Base model: MMLU accuracy (own CoT, greedy)", "base_accuracy"),
    ("RL model: MMLU accuracy (own CoT, greedy)", "rl_accuracy"),
    ("Base model: intact-CoT flip rate (control, prefill own full CoT)", "base_intact_flip"),
    ("RL model: intact-CoT flip rate (control, prefill own full CoT)", "rl_intact_flip"),
    ("Base model: truncated-CoT flip rate", "base_truncate_flip"),
    ("RL model: truncated-CoT flip rate", "rl_truncate_flip"),
    ("Base model: mismatched-CoT flip rate", "base_mismatch_flip"),
    ("RL model: mismatched-CoT flip rate", "rl_mismatch_flip"),
    ("Base model: no-CoT forced-answer flip rate", "base_nocot_flip"),
    ("RL model: no-CoT forced-answer flip rate", "rl_nocot_flip"),
]


def percentile(s, q):
    if not s:
        return float("nan")
    pos = q / 100.0 * (len(s) - 1)
    lo = int(pos); frac = pos - lo
    return s[lo] * (1 - frac) + s[lo + 1] * frac if lo + 1 < len(s) else s[lo]


def rate_ci(vals):
    n = len(vals)
    point = sum(vals) / n if n else float("nan")
    rng = random.Random(BOOT_SEED)
    boots = []
    for _ in range(BOOT_N):
        s = sum(vals[rng.randrange(n)] for _ in range(n)) / n
        boots.append(s)
    boots.sort()
    return point, percentile(boots, 2.5), percentile(boots, 97.5), n


def main():
    by = defaultdict(list)
    with open("eval_points.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                by[r["section"]].append((r["eval_order"], r["val"]))
    for k in by:
        by[k].sort()
    lines = []
    lines.append("# Self-CoT corruption flip rate: base vs outcome-only RL")
    lines.append("")
    lines.append("Model: Qwen2.5-1.5B-Instruct, GRPO with an outcome-only correctness reward on MMLU (no cue).")
    lines.append("Flip rate = fraction of items whose final answer changes when the model's own CoT is")
    lines.append("corrupted (truncated / replaced with another item's CoT / removed and the answer forced).")
    lines.append("Higher flip rate = the answer is more causally dependent on the CoT.")
    lines.append("Bootstrap: 2000 resamples, percentile 95% CI, seed 42.")
    lines.append("")
    for title, key in SECTIONS:
        vals = [v for _, v in by.get(key, [])]
        p, lo, hi, n = rate_ci(vals)
        lines.append("## %s" % title)
        lines.append("  rate = %.4f  (95%% CI %.4f-%.4f, n=%d)" % (p, lo, hi, n))
        lines.append("")
    print("\n".join(lines).rstrip("\n"))


if __name__ == "__main__":
    main()
