from lib.rewards import format_reward, expression_format_reward, equation_reward, get_think_and_answer


def test_get_think_and_answer():
    completion = "reasoning... </think><answer>1+2/3</answer>"
    think, answer = get_think_and_answer(completion)
    assert think == "reasoning... "
    assert answer == "1+2/3"

    completion = "reasoning...<think><think></think> </think><answer>1+2/3</answer>"
    think, answer = get_think_and_answer(completion)
    assert think is None and answer is None

    completion = "reasoning...<think>1+2/3</think><answer>1+2/3</answer>"
    think, answer = get_think_and_answer(completion)
    assert think is None and answer is None

    completion = "reasoning...<think>1+2/3<answer></think>1+2/3</answer>"
    think, answer = get_think_and_answer(completion)
    assert think is None and answer is None

    completion = "reasoning...</think><answer>123</answer><answer>456</answer>"
    think, answer = get_think_and_answer(completion)
    assert think is None and answer is None

def test_rewards():
    completions = [
        "reasoning... Wrong format</answer>",
        "reasoning...</think><answer>Wrong format</answer>",
        "reasoning...</think><answer>10+20</answer>",
        "reasoning...</think><answer>abc</answer>",
        "reasoning...</think><answer>1+2/3</answer>",
        "reasoning...</think><answer>2+3/1</answer>",
    ]
    nums = [2, 3, 1]
    target = ["5", "5", "5", "5", "5", "5"]
    assert format_reward(completions) == [0.0, 0.1, 0.1, 0.1, 0.1, 0.1]
    assert expression_format_reward(completions, nums) == [0.0, 0.0, 0.0, 0.0, 0.1, 0.1]
    assert equation_reward(completions, target, nums) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.8]
