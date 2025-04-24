# From https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/scripts/run_r1_grpo.py

import logging
import os

from unsloth import FastLanguageModel, is_bfloat16_supported

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import re

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

task_format = """<think>
[Your step-by-step reasoning]
</think>
<answer>
[Final answer (eg. 2 + 3 / 1)]
</answer>"""

# Reward functions


def get_think_and_answer(completion: str) -> tuple[str | None, str | None]:
    # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
    completion = "<think>" + completion

    # Check if the format is correct and extract the necessary parts
    regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\s*<answer>([\s\S]*?)<\/answer>$"
    match = re.fullmatch(regex, completion, re.DOTALL)
    if match is None or len(match.groups()) != 2:
        return None, None
    think, answer = match.groups()
    return think, answer


def eval_answer(answer: str, nums: list[int]) -> float | None:
    answer = answer.strip()

    # Check if the answer only contains numbers, operators, parentheses, and whitespace
    allowed_pattern = r"^[\d+\-*/().\s]+$"
    if not re.match(allowed_pattern, answer):
        return None

    # Check if the answer uses all the numbers exactly once
    used_numbers = [int(n) for n in re.findall(r"\d+", answer)]
    if sorted(used_numbers) != sorted(nums):
        return None

    try:
        # Evaluate the answer
        result = eval(answer, {"__builtins__": None}, {})
        return float(result)
    except Exception:
        # If the answer is not a valid expression, return None
        return None


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Checks if the completion is in the correct format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs

    Returns:
        list[float]: Reward scores
    """
    rewards = []
    for completion in completions:
        think, answer = get_think_and_answer(completion)
        correct = think is not None and answer is not None
        reward = 0.1 if correct else 0.0
        rewards.append(reward)

    return rewards


def expression_format_reward(completions: list[str], nums: list[int], **kwargs) -> list[float]:
    """
    Checks if the answer is a valid expression using only the numbers provided
    Args:
        completions (list[str]): Generated outputs
        nums (list[int]): Available numbers

    Returns:
        list[float]: Reward scores
    """
    rewards = []
    for completion in completions:
        think, answer = get_think_and_answer(completion)
        correct = answer is not None and eval_answer(answer, nums) is not None
        reward = 0.1 if correct else 0.0
        rewards.append(reward)

    return rewards


def equation_reward(completions: list[str], target: list[str], nums: list[int], **kwargs) -> list[float]:
    """
    Evaluates completions based on:
    2. Mathematical correctness of the answer

    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected number the expression should evaluate to
        nums (list[int]): Available numbers

    Returns:
        list[float]: Reward scores
    """

    rewards = []
    for completion, gt in zip(completions, target, strict=True):
        think, answer = get_think_and_answer(completion)
        reward = 0.0
        if answer is not None:
            result = eval_answer(answer, nums)
            if result is not None and abs(result - float(gt)) < 1e-5:
                reward = 0.8
        rewards.append(reward)

    return rewards


def log_completion(completions: list[str], target: list[str], nums: list[int], **kwargs) -> list[float]:
    format_correct = format_reward(completions[:1])[0] > 0
    expression_format_correct = expression_format_reward(completions[:1], nums[:1])[0] > 0
    equation_correct = equation_reward(completions[:1], target[:1], nums[:1])[0] > 0

    print("\n----------------")
    print(f"First completion:\n<think>{completions[0]}")
    print(f"Ground truth: {target[0]}")
    print(f"Numbers: {nums[0]}")
    print(f"Format correct?: {'Yes' if format_correct else 'No'}")
    print(f"Expression format correct?: {'Yes' if expression_format_correct else 'No'}")
    print(f"Equation correct?: {'Yes' if equation_correct else 'No'}", flush=True)
    return [0.0] * len(completions)


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
    def generate_r1_prompt(target, nums):
        r1_prefix = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Provide your step-by-step reasoning between <think> </think> tags and your final answer in <answer> </answer> tags. Example:\n\n"
                + task_format,
            },
            {
                "role": "user",
                "content": f"Using the numbers {nums}, create an expression that equals {target}. You can use basic arithmetic operations (+, -, *, /), but each number can only be used once.",
            },
            {"role": "assistant", "content": "<think>"},
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
        learning_rate=2e-7,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        adam_beta1=0.9,
        adam_beta2=0.99,
        max_grad_norm=40.0,
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
