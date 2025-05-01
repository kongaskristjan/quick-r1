from lib.rewards import equation_reward, expression_format_reward, format_reward, get_think_and_answer


def replace_newlines(s: str) -> str:
    return s.replace("\n", "<br>")


def log_completion(completions: list[str], target: list[int], nums: list[list[int]], **kwargs) -> list[float]:
    think, answer = get_think_and_answer(completions[0])
    format_correct = format_reward(completions[:1])[0] > 0
    expression_format_correct = expression_format_reward(completions[:1], nums[:1])[0] > 0
    equation_correct = equation_reward(completions[:1], target[:1], nums[:1])[0] > 0

    print("\n----------------")
    print(f"First completion:\n<think>{completions[0]}")
    if think is not None and answer is not None:
        print(f"Extracted thinking process: {replace_newlines(think[:10])}...{replace_newlines(think[-10:])}")  # Truncate for readability
        print(f"Extracted answer: {answer}")
    else:
        print("Thinking process or answer not found in the completion.")
    print(f"Ground truth: {target[0]}")
    print(f"Numbers: {nums[0]}")
    print(f"Format correct?: {'Yes' if format_correct else 'No'}")
    print(f"Expression format correct?: {'Yes' if expression_format_correct else 'No'}")
    print(f"Equation correct?: {'Yes' if equation_correct else 'No'}", flush=True)
    return [0.0] * len(completions)
