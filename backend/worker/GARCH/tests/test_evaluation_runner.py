from unittest.mock import Mock

from worker.GARCH.services import evaluation_runner as runner_module


def test_evaluate_distribution_uses_fit_with_retry_for_t(monkeypatch):
    garch = Mock()
    garch.fit_with_retry.return_value = {"alpha": 0.2}
    garch.generate_scenarios.return_value = ["scenario"]
    garch.validate_scenarios.return_value = {"ks": 0.1}

    monkeypatch.setattr(runner_module, "GARCHService", lambda: garch)

    runner = runner_module.EvaluationRunner(historical_data=[1, 2, 3])
    result = runner.evaluate_distribution("t", p=2, q=2, num_scenarios=10, horizon=20)

    garch.fit_with_retry.assert_called_once_with(historical_data=[1, 2, 3], p=2, q=2)
    garch._fit_once.assert_not_called()
    garch.generate_scenarios.assert_called_once_with(num_scenarios=10, horizon=20)
    garch.validate_scenarios.assert_called_once_with(["scenario"])
    assert result["distribution"] == "t"
    assert result["parameters"] == {"alpha": 0.2}
    assert result["metrics"] == {"ks": 0.1}


def test_evaluate_distribution_uses_fit_once_for_normal(monkeypatch):
    garch = Mock()
    garch._fit_once.return_value = {"beta": 0.7}
    garch.generate_scenarios.return_value = ["s1", "s2"]
    garch.validate_scenarios.return_value = {"acf": 0.2}

    monkeypatch.setattr(runner_module, "GARCHService", lambda: garch)

    runner = runner_module.EvaluationRunner(historical_data=[10, 20])
    result = runner.evaluate_distribution("normal")

    garch._fit_once.assert_called_once_with(
        historical_data=[10, 20], p=1, q=1, dist="normal"
    )
    garch.fit_with_retry.assert_not_called()
    assert result["num_scenarios"] == 500
    assert result["horizon"] == 252


def test_compare_distributions_collects_errors():
    runner = runner_module.EvaluationRunner(historical_data=[1])

    def fake_evaluate(dist):
        if dist == "bad":
            raise RuntimeError("broken")
        return {"distribution": dist}

    runner.evaluate_distribution = fake_evaluate
    results = runner.compare_distributions(["normal", "bad"])

    assert results["normal"] == {"distribution": "normal"}
    assert results["bad"] == {"error": "broken"}
