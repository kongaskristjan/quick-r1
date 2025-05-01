# From https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/scripts/run_r1_grpo.py

import os

from unsloth import FastLanguageModel, is_bfloat16_supported

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

from lib.logging import log_completion
from lib.rewards import equation_reward, expression_format_reward, format_reward


def main() -> None:
    lora_rank = 256  # Larger rank = smarter, but slower. Suggested 8, 16, 32, 64, 128

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_seq_length=1024,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.6,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Remove QKVO if out of memory
        lora_alpha=lora_rank,
        use_rslora=True,
        use_gradient_checkpointing="unsloth",  # Enable long context finetuning
        random_state=3407,
    )

    # Load dataset from Hugging Face Hub
    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    # select a random subset of 50k samples
    dataset = dataset.shuffle(seed=42).select(range(50000))

    # Generate r1 prompt with a prefix for the model to already start with the thinking process
    def generate_r1_prompt(target: str, nums: list[int]):
        r1_prefix = [
            {
                "role": "system",
                "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.",
            },
            {
                "role": "user",
                "content": f"Using the numbers {nums}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. Think step by step inside <think> tags.",
            },
            {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
        ]
        return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), "target": target, "nums": nums}

    # convert our dataset to the r1 prompt
    dataset = dataset.map(lambda x: generate_r1_prompt(x["target"], x["nums"]))

    # split the dataset into train and test
    train_test_split = dataset.train_test_split(test_size=0.1)

    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    # Hyperparameters
    training_args = GRPOConfig(
        output_dir="qwen-r1-aha-moment",
        learning_rate=2e-6,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        adam_beta1=0.9,
        adam_beta2=0.99,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        logging_steps=1,
        max_steps=450,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # GRPO specific parameters
        max_prompt_length=256,
        max_completion_length=1024,  # max length of the generated output for our solution
        num_generations=8,
        beta=0.05,
    )

    # Training loop
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward, expression_format_reward, equation_reward, log_completion],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    train_result = trainer.train()

    # Save model
    model.save_lora("grpo_saved_lora")
    trainer.save_model(training_args.output_dir)

    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
