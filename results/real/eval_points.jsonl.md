# eval_points.jsonl — sidecar (REPRO_CONTRACT)

Generated-By: src/eval_corruption.py (checkpoints from src/train_grpo.py)
Command: python3 src/eval_corruption.py --base_adapter runs/rl/checkpoint-0 --rl_adapter runs/rl/checkpoint-final --n_eval 300 --seed 42
Git-Commit: 12da608880867be46ae58a5e53a78ef6e4e248d9
Seeds: 42 (MMLU eval-slice sampling in src/task.py; greedy decoding so generation is deterministic; 2000-resample percentile bootstrap for CIs; GRPO training seed 0)
Source-Data: MMLU (cais/mmlu) reserved eval slice via src/task.py load_items('mmlu','eval'); model Qwen2.5-1.5B-Instruct adapted by outcome-only GRPO (reward=answer-correctness, no cue) to runs/rl/checkpoint-0 (base anchor, pre-RL) and runs/rl/checkpoint-final (after 120 steps), RTX 5090, 2026-06-24, torch 2.12 cu130, trl 1.5.1
Analysis-Command: cd results/real && python3 recompute.py  (each section rate = mean(val) with 95% bootstrap CI)
Columns:
  section (e.g. base_mismatch_flip, rl_accuracy); eval_order (position within section); val (0/1: for *_flip = did the final answer change under that corruption; for *_accuracy = was the own-CoT answer correct)
