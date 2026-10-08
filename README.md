# LoRA / QLoRA Fine-Tuning + GRPO Post-Training

A reproducible mathematical-reasoning post-training project covering **parameter-efficient fine-tuning (LoRA/QLoRA)** and **GRPO reinforcement-learning-based post-training**.

The original project explored DeepSeek-Math-7B and Qwen2.5-Math-7B on mathematical reasoning data including MATH, GSM8K, NuminaMath, and AoPS. This repository now also contains explicit training scripts so the workflow is inspectable instead of being represented only by a project description.

## What this repository demonstrates

- 4-bit QLoRA configuration with NF4 quantization.
- LoRA adapters for causal language models.
- Supervised fine-tuning with TRL SFTTrainer.
- GRPO post-training with explicit reward functions.
- Mathematical-answer extraction and correctness rewards.
- Dataset formatting for reasoning tasks.
- Reproducible CLI configuration.
- Lightweight unit tests for the deterministic reward logic.

## Files

~~~text
train_qlora.py          QLoRA supervised fine-tuning entry point
train_grpo.py           GRPO post-training entry point
rewards.py              deterministic reward / answer-normalization logic
tests/test_rewards.py   unit tests for reward functions
requirements.txt        Python dependencies
README.md               project documentation
~~~

## QLoRA example

~~~bash
python train_qlora.py \
  --model-id Qwen/Qwen2.5-Math-7B-Instruct \
  --dataset-id openai/gsm8k \
  --dataset-config main \
  --output-dir outputs/qlora
~~~

For a small smoke run before committing substantial GPU time:

~~~bash
python train_qlora.py --max-samples 64 --epochs 0.05
~~~

## GRPO example

~~~bash
python train_grpo.py \
  --model-id Qwen/Qwen2.5-Math-7B-Instruct \
  --dataset-id openai/gsm8k \
  --dataset-config main \
  --output-dir outputs/grpo
~~~

The GRPO script uses two transparent rewards:

1. **correctness_reward** - compares the final extracted numeric answer with the reference answer.
2. **format_reward** - provides a small shaping reward when the completion clearly marks its final answer.

## Tests

~~~bash
python -m unittest discover -s tests -v
~~~

The unit tests validate the deterministic reward and answer-normalization layer. They do **not** claim model-quality improvement; that must be established from actual training/evaluation runs.

## Hardware note

7B QLoRA/GRPO training is GPU-intensive. Use an appropriate CUDA-capable environment and reduce sample count / sequence length for smoke tests. Quantization reduces memory requirements but does not eliminate the need for adequate GPU resources.

## Evidence policy

This repository intentionally does not publish invented accuracy or benchmark improvements. Any future performance claim should be accompanied by the model/configuration, dataset split, seed, evaluation procedure, and raw result artifact.

## Project URL

https://github.com/ivin-santhosh/LoRA-QLoRA-Fine-Tuning
