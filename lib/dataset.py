from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


def _generate_prompt(target: int, nums: list[int], tokenizer: AutoTokenizer) -> dict:
    assert isinstance(target, int)
    assert all(isinstance(n, int) for n in nums)
    nums_formatted = ", ".join(str(n) for n in nums)
    r1_prefix = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.",
        },
        {
            "role": "user",
            "content": f"Using the numbers {nums_formatted}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. Think step by step inside <think> tags.",
        },
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]
    return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), "target": target, "nums": nums}


def load_countdown_dataset(tokenizer: AutoTokenizer, size: int = 50000) -> tuple[Dataset, Dataset]:
    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    dataset = dataset.shuffle(seed=42).select(range(size))
    dataset = dataset.map(lambda x: _generate_prompt(x["target"], x["nums"], tokenizer))
    train_test_split = dataset.train_test_split(test_size=0.1)

    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    return train_dataset, test_dataset
