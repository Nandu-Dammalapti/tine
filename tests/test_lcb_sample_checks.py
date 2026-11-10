import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "lcb_sample_checks", Path(__file__).resolve().parents[1] / "scripts" / "lcb_sample_checks.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def test_atcoder_samples_with_explanations_and_empty_output():
    query = ("Print the answer.Sample Input 1\n\n3 5\r\nxooox\n\nSample Output 1\n\n2\r\n\n"
             "All the people are free.\n\nSample Input 2\n\n2\n()\n\nSample Output 2\n\n\n\n"
             "The string may be empty.\n\nSample Input 3\n\n1\nab\n\nSample Output 3\n\nab\ncd")
    samples = checks.stdin_samples(query)
    assert samples == [
        {"input": "3 5\nxooox\n", "output": "2"},
        {"input": "2\n()\n", "output": ""},
        {"input": "1\nab\n", "output": "ab\ncd"},
    ]


def test_codeforces_samples_drop_blank_input_lines():
    query = "Sample Input 1:\n6\n\nabc\n\nacb\n\n\n\nSample Output 1:\n\nYES\nNO\n\n\nNote\n\nfirst case"
    assert checks.stdin_samples(query) == [{"input": "6\nabc\nacb\n", "output": "YES\nNO"}]


def test_functional_examples_parse_literals():
    query = ('Example 1:\n\nInput: nums = [1,2,3], target = 4, flag = true\nOutput: [0,2]\n'
             'Explanation: ...\n\nExample 2:\n\nInput: s = "a, b"\nOutput: null\n')
    prompt = "class Solution:\n    def solve(self, nums: List[int], target: int) -> List[int]:\n"
    method, samples = checks.functional_samples(query, prompt)
    assert method == "solve"
    assert len(samples) == 2
    assert eval(samples[0]["kwargs"]) == {"nums": [1, 2, 3], "target": 4, "flag": True}
    assert eval(samples[0]["expected"]) == [0, 2]
    assert eval(samples[1]["kwargs"]) == {"s": "a, b"} and samples[1]["expected"] == "None"


def test_functional_examples_skip_unparseable_inputs():
    query = "Example 1:\n\nInput: [1, 2]\nOutput: 3\n\nExample 2:\n\nInput: x = foo(1)\nOutput: 2\n"
    method, samples = checks.functional_samples(query, "def f(self, x):")
    assert method == "f" and samples == []


def test_token_comparison():
    assert checks.tokens_match("YES\n1 2\n", "yes 1 2")
    assert checks.tokens_match("0.3333333", "0.33333331")
    assert not checks.tokens_match("1 2", "1 2 3")
    assert not checks.tokens_match("1.5", "1.6")
    assert checks.tokens_match("\n\n", "")


def test_preamble_keeps_three_argument_pow():
    namespace = {}
    exec(checks.PREAMBLE, namespace)
    assert namespace["pow"](2, 10, 1000) == 24
