import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import beta, binom

from tine import planning
from tine.calibration import clopper_pearson_upper, minimum_risk_tasks
from tine.planning import (
    certify_tasks,
    failure_steps,
    max_failures,
    pass_probability,
    plan,
    rate_estimate,
    sequence_pass_bounds,
    split_pass_probability,
    tasks_for_power,
    tasks_needed,
    total_tasks_for_power,
)

spec = importlib.util.spec_from_file_location(
    "tine_plan_certification", Path(__file__).parents[1] / "scripts" / "plan_certification.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)

GRID = [(0.01, 0.05), (0.02, 0.05), (0.03, 0.05), (0.04, 0.05), (0.05, 0.1), (0.1, 0.01)]


def brute_max_failures(n, epsilon, delta):
    """Reference written straight from scipy: the bound is the beta quantile."""
    passing = [k for k in range(n) if beta.isf(delta, k + 1, n - k) <= epsilon]
    return max(passing) if passing else None


def test_documented_examples_are_exact():
    # results/2026-11-14-api-judge: 82 certify tasks at a 4% limit, then 137.
    assert clopper_pearson_upper(0, 82, 0.05) == pytest.approx(0.035874, abs=5e-7)
    assert max_failures(82, 0.04, 0.05) == 0
    assert clopper_pearson_upper(1, 82, 0.05) > 0.04
    assert max_failures(137, 0.04, 0.05) == 1
    assert clopper_pearson_upper(1, 137, 0.05) == pytest.approx(0.034157, abs=5e-7)
    assert tasks_needed(0.02, 0.05) == 149
    assert max_failures(148, 0.02, 0.05) is None
    assert [tasks_needed(0.04, 0.05, k) for k in range(4)] == [74, 117, 156, 192]
    assert [tasks_needed(0.02, 0.05, k) for k in range(4)] == [149, 236, 313, 386]


@pytest.mark.parametrize("epsilon,delta", GRID)
def test_max_failures_matches_scipy(epsilon, delta):
    for n in (1, 5, 30, 82, 137, 149, 274, 500, 1000):
        allowed = max_failures(n, epsilon, delta)
        assert allowed == brute_max_failures(n, epsilon, delta)
        if allowed is not None:
            # The same statement as a binomial tail: k passes when P(X <= k | epsilon) <= delta.
            assert binom.cdf(allowed, n, epsilon) <= delta
            assert binom.cdf(allowed + 1, n, epsilon) > delta


@pytest.mark.parametrize("epsilon,delta", GRID)
def test_tasks_needed_is_the_first_passing_size(epsilon, delta):
    for failures in (0, 1, 2, 5):
        n = tasks_needed(epsilon, delta, failures)
        assert clopper_pearson_upper(failures, n, delta) <= epsilon
        assert clopper_pearson_upper(failures, n - 1, delta) > epsilon
        assert max_failures(n, epsilon, delta) == failures
    assert tasks_needed(epsilon, delta, 0) == minimum_risk_tasks(epsilon, delta, 1)


def test_tasks_needed_agrees_with_minimum_risk_tasks_widely():
    for epsilon in np.linspace(0.003, 0.6, 40):
        for delta in (0.001, 0.01, 0.05, 0.1, 0.3):
            assert tasks_needed(float(epsilon), delta) == minimum_risk_tasks(float(epsilon), delta, 1)


def test_monotonicity():
    allowed = [max_failures(n, 0.04, 0.05) for n in range(74, 600)]
    assert all(a <= b for a, b in zip(allowed, allowed[1:]))
    needed = [tasks_needed(0.04, 0.05, k) for k in range(12)]
    assert all(a < b for a, b in zip(needed, needed[1:]))
    # A tighter limit or more confidence never needs fewer tasks.
    assert tasks_needed(0.02, 0.05) > tasks_needed(0.04, 0.05)
    assert tasks_needed(0.04, 0.01) > tasks_needed(0.04, 0.05)
    # A worse chain never passes more often.
    chances = [pass_probability(300, 0.04, 0.05, rate) for rate in np.linspace(0, 0.1, 30)]
    assert all(a >= b for a, b in zip(chances, chances[1:]))
    powered = [tasks_for_power(0.04, 0.05, rate, stable=True) for rate in (0.0, 0.005, 0.01, 0.02, 0.03)]
    assert all(a <= b for a, b in zip(powered, powered[1:]))


def test_failure_steps_describe_the_whole_test():
    steps = failure_steps(0.04, 0.05, 400)
    assert steps == [tasks_needed(0.04, 0.05, k) for k in range(len(steps))]
    assert tasks_needed(0.04, 0.05, len(steps)) > 400
    for n in range(0, 401, 7):
        expected = sum(step <= n for step in steps) - 1
        assert max_failures(n, 0.04, 0.05) == (expected if expected >= 0 else None)
    assert failure_steps(0.04, 0.05, 73) == []
    assert failure_steps(0.04, 0.05, 400) == steps  # cached answers do not change


def test_pass_probability_matches_scipy():
    assert pass_probability(137, 0.04, 0.05, 0.01) == pytest.approx(binom.cdf(1, 137, 0.01))
    assert pass_probability(82, 0.04, 0.05, 0.01) == pytest.approx(0.99 ** 82)
    assert pass_probability(300, 0.02, 0.05, 0.007) == pytest.approx(binom.cdf(1, 300, 0.007))
    # The guarantee: a chain at or over the limit passes with probability at most delta.
    for n in (74, 117, 137, 500, 2000):
        assert pass_probability(n, 0.04, 0.05, 0.04) <= 0.05
        assert pass_probability(n, 0.04, 0.05, 0.06) <= 0.05


def test_edge_cases():
    assert max_failures(0, 0.04, 0.05) is None
    assert pass_probability(0, 0.04, 0.05, 0.0) == 0.0
    assert pass_probability(73, 0.04, 0.05, 0.0) == 0.0
    assert pass_probability(74, 0.04, 0.05, 0.0) == 1.0
    assert pass_probability(74, 0.04, 0.05, 1.0) == 0.0
    assert tasks_needed(0.0, 0.05) is None
    assert max_failures(10_000, 0.0, 0.05) is None
    assert tasks_needed(1.0, 0.05) == 1 and tasks_needed(1.0, 0.05, 3) == 3
    assert tasks_needed(0.04, 0.05, 0, max_tasks=73) is None
    assert tasks_needed(0.04, 0.05, 0, max_tasks=74) == 74
    assert tasks_for_power(0.04, 0.05, 0.0) == 74
    assert tasks_for_power(0.04, 0.05, 0.0, stable=True) == 74
    assert tasks_for_power(0.04, 0.05, 0.04) is None
    assert tasks_for_power(0.04, 0.05, 0.2) is None
    assert total_tasks_for_power(0.3, 0.04, 0.05, 0.04) is None
    assert total_tasks_for_power(0.0, 0.04, 0.05, 0.01) is None
    assert tasks_for_power(0.02, 0.05, 0.01, max_tasks=500) is None
    for bad in (-0.1, 1.5, "x", True):
        with pytest.raises(ValueError):
            max_failures(10, bad, 0.05)
    with pytest.raises(ValueError):
        max_failures(-1, 0.04, 0.05)
    with pytest.raises(ValueError):
        tasks_needed(0.04, 0.0)
    with pytest.raises(ValueError):
        tasks_for_power(0.04, 0.05, 0.01, power=1.0)
    with pytest.raises(ValueError):
        plan(100, 0.3, 0.04, split="stratified")


@pytest.mark.parametrize("epsilon,rate,power", [(0.04, 0.01, 0.8), (0.04, 0.0073, 0.8), (0.02, 0.0073, 0.8),
                                                (0.03, 0.005, 0.9), (0.05, 0.03, 0.5), (0.04, 0.02, 0.95)])
def test_tasks_for_power_against_brute_force(epsilon, rate, power):
    first = tasks_for_power(epsilon, 0.05, rate, power)
    stable = tasks_for_power(epsilon, 0.05, rate, power, stable=True)
    limit = stable + 600
    steps = np.asarray(failure_steps(epsilon, 0.05, limit))
    sizes = np.arange(steps[0], limit + 1)
    chances = binom.cdf(np.searchsorted(steps, sizes, side="right") - 1, sizes, rate)
    reached = chances >= power
    assert first == sizes[np.argmax(reached)]
    assert pass_probability(first, epsilon, 0.05, rate) >= power
    assert reached[sizes >= stable].all()
    assert first <= stable
    if stable > first:
        assert not reached[sizes == stable - 1][0]


def test_power_is_not_monotone_in_tasks_and_stable_says_so():
    # Two failures are allowed from 313 tasks; at 312 only one is.
    assert pass_probability(236, 0.02, 0.05, 0.0033) > pass_probability(312, 0.02, 0.05, 0.0033)
    assert tasks_for_power(0.02, 0.05, 0.0033) == 236
    assert tasks_for_power(0.02, 0.05, 0.0033, stable=True) == 313
    assert pass_probability(312, 0.02, 0.05, 0.0033) < 0.8 <= pass_probability(313, 0.02, 0.05, 0.0033)


def test_sequence_pass_bounds():
    assert sequence_pass_bounds([0.9]) == pytest.approx((0.9, 0.9))
    low, high = sequence_pass_bounds([0.9, 0.95, 0.8])
    assert (low, high) == pytest.approx((0.65, 0.8))
    assert sequence_pass_bounds([0.3, 0.3, 0.3])[0] == 0.0
    with pytest.raises(ValueError):
        sequence_pass_bounds([])


def test_rate_estimate():
    estimate = rate_estimate(2, 274)
    assert estimate["rate"] == pytest.approx(2 / 274)
    assert estimate["upper"] == pytest.approx(beta.ppf(0.95, 3, 272))
    assert estimate["lower"] == pytest.approx(beta.ppf(0.05, 2, 273))
    assert rate_estimate(0, 50)["lower"] == 0.0
    with pytest.raises(ValueError):
        rate_estimate(3, 2)
    with pytest.raises(ValueError):
        rate_estimate(0, 0)


def test_certify_tasks_and_plan():
    assert certify_tasks(274, 0.3) == 82 and certify_tasks(274, 0.5) == 137
    assert certify_tasks(10, 0.3) == 3 and certify_tasks(0, 0.5) == 0 and certify_tasks(7, 1.0) == 7
    default = plan(274, 0.3, 0.04, true_rate=2 / 274)
    assert default["certify_tasks"] == 82 and default["max_failures"] == 0
    assert default["zero_failure_bound"] == pytest.approx(0.035874, abs=5e-7)
    assert default["pass_probability"] == pytest.approx((1 - 2 / 274) ** 82)
    half = plan(274, 0.5, 0.04, true_rate=2 / 274)
    assert half["max_failures"] == 1
    assert half["bound_at_max_failures"] == pytest.approx(0.034157, abs=5e-7)
    assert half["pass_probability"] == pytest.approx(binom.cdf(1, 137, 2 / 274))
    strict = plan(274, 0.5, 0.02, true_rate=2 / 274)
    assert strict["max_failures"] is None and strict["bound_at_max_failures"] is None
    assert strict["pass_probability"] == 0.0 and strict["zero_failure_tasks"] == 149
    assert plan(274, 0.5, 0.04)["pass_probability"] is None
    assert plan(0, 0.5, 0.04, true_rate=0.01)["pass_probability"] == 0.0


def test_random_split_is_the_average_over_certify_sizes():
    total, fraction, rate = 120, 0.5, 0.01
    expected = sum(binom.pmf(n, total, fraction) * pass_probability(n, 0.04, 0.05, rate)
                   for n in range(total + 1))
    assert split_pass_probability(total, fraction, 0.04, 0.05, rate, split="random") == pytest.approx(expected)
    assert plan(total, fraction, 0.04, true_rate=rate, split="random")["pass_probability"] == pytest.approx(expected)
    # A fixed split of 274 at one half cannot reach 149 certify tasks; a random one sometimes does.
    assert split_pass_probability(274, 0.5, 0.02, 0.05, 0.0, split="fixed") == 0.0
    assert split_pass_probability(274, 0.5, 0.02, 0.05, 0.0, split="random") == pytest.approx(
        binom.sf(148, 274, 0.5))
    assert split_pass_probability(50, 0.5, 0.04, 0.05, 0.0, split="random") == 0.0


@pytest.mark.parametrize("total,fraction", [(3000, 0.03), (100_000, 0.001), (80, 0.999), (74, 1.0)])
def test_random_split_keeps_all_but_the_far_tails(total, fraction):
    # A small or large certify share gives a skewed size; the average must still be the full sum.
    exact = sum(binom.pmf(n, total, fraction) * pass_probability(n, 0.04, 0.05, 0.01)
                for n in range(74, min(total, 400) + 1))
    assert split_pass_probability(total, fraction, 0.04, 0.05, 0.01, split="random") == pytest.approx(
        exact, rel=1e-12, abs=1e-14)


def test_random_split_matches_simulation_of_the_cascade_split():
    # run_cascade_experiment draws each task's part independently; check the model of it.
    rng = np.random.default_rng(0)
    total, rate, runs = 400, 0.01, 20_000
    sizes = rng.binomial(total, 0.3, size=runs)
    failures = rng.binomial(sizes, rate)
    steps = np.asarray(failure_steps(0.04, 0.05, total))
    passed = failures <= np.searchsorted(steps, sizes, side="right") - 1
    exact = split_pass_probability(total, 0.3, 0.04, 0.05, rate, split="random")
    assert passed.mean() == pytest.approx(exact, abs=4 * np.sqrt(exact * (1 - exact) / runs))


@pytest.mark.parametrize("fraction", [0.3, 0.5])
def test_total_tasks_for_power_inverts_plan(fraction):
    rate = 2 / 274
    for stable in (False, True):
        needed = tasks_for_power(0.04, 0.05, rate, stable=stable)
        total = total_tasks_for_power(fraction, 0.04, 0.05, rate, stable=stable)
        assert certify_tasks(total, fraction) >= needed > certify_tasks(total - 1, fraction)
        assert plan(total, fraction, 0.04, true_rate=rate)["pass_probability"] >= 0.8
    total = total_tasks_for_power(fraction, 0.04, 0.05, rate, split="random", stable=False)
    assert split_pass_probability(total, fraction, 0.04, 0.05, rate, split="random") >= 0.8
    assert split_pass_probability(total - 1, fraction, 0.04, 0.05, rate, split="random") < 0.8
    settled = total_tasks_for_power(fraction, 0.04, 0.05, rate, split="random")
    assert settled >= total
    later = [split_pass_probability(n, fraction, 0.04, 0.05, rate, split="random")
             for n in range(settled, settled + 400, 7)]
    assert min(later) >= 0.8
    with pytest.raises(ValueError):
        total_tasks_for_power(fraction, 0.04, 0.05, rate, split="other")


def test_project_numbers_in_the_doc():
    """docs/PLANNING.md quotes these; a change here means the doc must change."""
    rate = 2 / 274
    certify = {eps: tasks_for_power(eps, 0.05, rate, stable=True) for eps in (0.02, 0.03, 0.04)}
    assert certify == {0.02: 655, 0.03: 257, 0.04: 156}
    totals = {(eps, f): total_tasks_for_power(f, eps, 0.05, rate) for eps in certify for f in (0.3, 0.5)}
    assert totals == {(0.02, 0.3): 2184, (0.02, 0.5): 1310, (0.03, 0.3): 857, (0.03, 0.5): 514,
                      (0.04, 0.3): 520, (0.04, 0.5): 312}
    assert tasks_for_power(0.02, 0.05, rate_estimate(2, 274)["upper"]) is None
    # All 500 SWE-bench Verified tasks at a 2% limit, 30/50/20: quoted as 45% (fixed), 42% (random).
    assert pass_probability(250, 0.02, 0.05, rate) == pytest.approx(binom.cdf(1, 250, rate))
    assert round(100 * pass_probability(250, 0.02, 0.05, rate)) == 45
    assert round(100 * split_pass_probability(500, 0.5, 0.02, 0.05, rate, split="random")) == 42
    # Worst case with 49 earlier chains of the same rate: each needs 1 - 0.2 / 50.
    assert [tasks_for_power(eps, 0.05, rate, 1 - 0.2 / 50, stable=True)
            for eps in (0.02, 0.03, 0.04)] == [1508, 604, 358]


def test_power_search_against_an_independent_table():
    """The whole test rebuilt from binomial tails alone, with no bound function and no steps."""
    rate, allowed, chances = 0.01, -1, [0.0]
    for n in range(1, 1600):
        while binom.cdf(allowed + 1, n, 0.03) <= 0.05:  # k passes when P(X <= k | epsilon) <= delta
            allowed += 1
        assert max_failures(n, 0.03, 0.05) == (allowed if allowed >= 0 else None)
        chances.append(float(binom.cdf(allowed, n, rate)) if allowed >= 0 else 0.0)
    reached = np.asarray(chances) >= 0.8
    assert tasks_for_power(0.03, 0.05, rate) == int(np.argmax(reached)) == 303
    assert tasks_for_power(0.03, 0.05, rate, stable=True) == int(np.flatnonzero(~reached)[-1]) + 1 == 392


def test_cli_prints_a_table(capsys):
    cli.main(["--epsilon", "0.04", "0.02", "--observed", "2", "274", "--total", "274",
              "--split", "40/30/30", "30/50/20"])
    out = capsys.readouterr().out
    assert "2 of 274" in out and "0.73%" in out and "2.28%" in out
    assert "| 0.04 | 82 | 0 | 3.59% |" in out
    assert "| 0.04 | 137 | 1 | 2.16% |" in out
    assert "| 0.02 | 137 | none | 2.16% |" in out
    assert "cannot" in out
    assert cli.parse_split("40/30/30") == pytest.approx(0.3)
    assert cli.parse_split("0.5") == pytest.approx(0.5)
    for bad in ("40/30", "40/30/20", "a/b/c", "1.5"):
        with pytest.raises(Exception):
            cli.parse_split(bad)


@pytest.mark.parametrize("arguments", [["--power", "1.0"], ["--epsilon", "1.5"], ["--delta", "0"],
                                       ["--observed", "5", "3"], ["--rate", "-0.1"], ["--total", "-5"],
                                       ["--split", "40/30/20"], ["--earlier-chains", "-1"]])
def test_cli_rejects_bad_numbers_before_printing(arguments, capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(arguments)
    assert stopped.value.code == 2
    assert capsys.readouterr().out == ""


def test_cli_says_the_figures_are_lower_limits(capsys):
    cli.main(["--epsilon", "0.04", "--rate", "0.01", "--certify", "137"])
    assert "lower limits" in capsys.readouterr().out


def test_cli_with_a_stated_rate_and_no_total(capsys):
    cli.main(["--epsilon", "0.04", "--rate", "0.01", "--certify", "137"])
    out = capsys.readouterr().out
    assert "1.00%" in out and "| 0.04 | 137 | 1 |" in out
    assert planning.pass_probability(137, 0.04, 0.05, 0.01) == pytest.approx(binom.cdf(1, 137, 0.01))
