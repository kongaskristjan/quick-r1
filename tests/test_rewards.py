from lib.rewards import format_reward, expression_format_reward, equation_reward


def test_rewards():
    completions = [
        "reasoning... Wrong format</answer>",
        "reasoning...</think><answer>Wrong format</answer>",
        "reasoning...</think><answer>1+2/3</answer>",
        "reasoning...</think><answer>2+3/1</answer>",
    ]
    nums = [2, 3, 1]
    target = ["5", "5", "5", "5"]
    assert format_reward(completions) == [0.0, 0.1, 0.1, 0.1]
    assert expression_format_reward(completions, nums) == [0.0, 0.0, 0.1, 0.1]
    assert equation_reward(completions, target, nums) == [0.0, 0.0, 0.0, 0.8]
