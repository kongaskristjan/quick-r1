# From https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/scripts/run_r1_grpo.py

import os

from unsloth import FastLanguageModel, is_bfloat16_supported

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from trl import GRPOConfig, GRPOTrainer

from lib.dataset import load_countdown_dataset
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

    train_dataset, test_dataset = load_countdown_dataset(tokenizer)

    # Hyperparameters
    training_args = GRPOConfig(
        output_dir="qwen-r1-aha-moment",
        learning_rate=1e-6,
        optim="adamw_8bit",
        adam_beta1=0.9,
        adam_beta2=0.99,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        logging_steps=1,
        max_steps=450,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # GRPO specific parameters
        max_prompt_length=256,
        max_completion_length=1024,  # max length of the generated output for our solution
        num_generations=4,
        beta=0.05,
        #        scale_rewards=False,
        #        loss_type="dr_grpo",
        #        mask_truncated_completions=True,
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
