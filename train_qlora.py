from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer


@dataclass(frozen=True)
class TrainConfig:
    model_id: str
    dataset_id: str
    dataset_config: str | None
    split: str
    output_dir: str
    max_samples: int | None
    max_length: int
    epochs: float
    learning_rate: float
    seed: int


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="QLoRA supervised fine-tuning for mathematical reasoning models.")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-Math-7B-Instruct")
    p.add_argument("--dataset-id", default="openai/gsm8k")
    p.add_argument("--dataset-config", default="main")
    p.add_argument("--split", default="train")
    p.add_argument("--output-dir", default="outputs/qlora")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    return TrainConfig(
        model_id=a.model_id,
        dataset_id=a.dataset_id,
        dataset_config=(a.dataset_config or None),
        split=a.split,
        output_dir=a.output_dir,
        max_samples=a.max_samples,
        max_length=a.max_length,
        epochs=a.epochs,
        learning_rate=a.learning_rate,
        seed=a.seed,
    )


def choose_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    compute_dtype = choose_compute_dtype()
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    dataset = load_dataset(cfg.dataset_id, cfg.dataset_config, split=cfg.split)
    if cfg.max_samples:
        dataset = dataset.select(range(min(cfg.max_samples, len(dataset))))

    def format_example(example: dict) -> dict:
        question = str(example.get("question", example.get("problem", ""))).strip()
        answer = str(example.get("answer", example.get("solution", ""))).strip()
        messages = [
            {"role": "system", "content": "Solve the problem carefully and show concise reasoning before the final answer."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    dataset = dataset.map(format_example, desc="Formatting training examples")

    training_args = SFTConfig(
        output_dir=cfg.output_dir,
        dataset_text_field="text",
        max_length=cfg.max_length,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        gradient_checkpointing=True,
        fp16=(compute_dtype == torch.float16),
        bf16=(compute_dtype == torch.bfloat16),
        report_to="none",
        seed=cfg.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()
