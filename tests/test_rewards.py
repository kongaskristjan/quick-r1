from lib.rewards import _extract_between, _find_all, equation_reward, expression_format_reward, format_reward, get_think_and_answer


def rewards_to_correct(rewards: list[float]) -> list[bool]:
    return [reward > 0.0 for reward in rewards]


def test_find_all():
    completion = "<think>reasoning... </think><answer>1+2/3</answer>"
    indices = _find_all(completion, "</think>")
    assert indices == [completion.find("</think>")]


def test_extract_between():
    completion = "<think>reasoning... </think> - <answer>1+2/3</answer>"
    extracts = _extract_between(completion, ["<think>", "</think>", "<answer>", "</answer>"])
    assert extracts == ["", "reasoning... ", " - ", "1+2/3", ""]

    assert _extract_between("<tag>", ["<tag>"]) is not None
    assert _extract_between("<tag><tag>", ["<tag>"]) is None
    assert _extract_between("<tag>", ["<tag>", "</tag>"]) is None


def test_get_think_and_answer():
    completion = "reasoning... </think> \n <answer>1+2/3</answer>"
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


def test_format_reward():
    completions = [
        "reasoning... Wrong format</answer>",
        "reasoning...</think><answer>Correct format</answer>",
    ]
    assert rewards_to_correct(format_reward(completions)) == [False, True]


def test_expression_format_reward():
    completions = [
        "reasoning... Wrong format</answer>",
        "reasoning...</think><answer>Wrong expression format</answer>",
        "reasoning...</think><answer>1 + 2 / 3 = 5</answer>",
        "reasoning...</think><answer> 1 + 2 / 3 </answer>",
        "reasoning...</think><answer> (59 - 29) + 21 </answer>",
    ]
    nums = [[59, 21, 29]] * len(completions)
    assert rewards_to_correct(expression_format_reward(completions, nums)) == [False, False, False, False, True]


def test_equation_reward():
    completions = [
        "reasoning... Wrong format</answer>",
        "reasoning...</think><answer>Wrong equation format</answer>",
        "reasoning...</think><answer>1 / 3 + 2</answer>",
        "reasoning...</think><answer>2 + 3 / 1</answer>",
    ]

    target = [5] * len(completions)
    nums = [[1, 2, 3]] * len(completions)
    assert rewards_to_correct(equation_reward(completions, target, nums)) == [False, False, False, True]
