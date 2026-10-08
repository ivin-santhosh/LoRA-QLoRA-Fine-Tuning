from __future__ import annotations

import argparse

from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from rewards import correctness_reward, format_reward


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO post-training for mathematical reasoning.")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-Math-7B-Instruct")
    p.add_argument("--dataset-id", default="openai/gsm8k")
    p.add_argument("--dataset-config", default="main")
    p.add_argument("--split", default="train")
    p.add_argument("--output-dir", default="outputs/grpo")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--max-completion-length", type=int, default=512)
    p.add_argument("--num-generations", type=int, default=2)
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_device_batch_size % args.num_generations != 0:
        raise ValueError("--per-device-batch-size must be divisible by --num-generations for a single-process run.")

    dataset = load_dataset(args.dataset_id, args.dataset_config or None, split=args.split)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    def add_prompt(example: dict) -> dict:
        question = str(example.get("question", example.get("problem", ""))).strip()
        return {"prompt": f"Solve this problem carefully. End with a clearly marked final answer.\\n\\n{question}"}

    dataset = dataset.map(add_prompt, desc="Preparing GRPO prompts")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=4,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        gradient_checkpointing=True,
        report_to="none",
        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=args.model_id,
        reward_funcs=[correctness_reward, format_reward],
        args=config,
        train_dataset=dataset,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
