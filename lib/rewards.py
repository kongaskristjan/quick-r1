import re

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
