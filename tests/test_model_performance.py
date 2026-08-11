"""Performance and validity tests for the four models and the ensemble.

Run on a synthetic cohort with known ground truth: 'active' indicators keep
appearing through the end of the window, 'dormant' ones went quiet weeks ago.
A sound model must rank active well above dormant.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score


@pytest.fixture(scope="module")
def scored(noi, synthetic_panel):
    """Full scoring pipeline (pre-formatting) on the synthetic cohort."""
    features = noi.build_features(synthetic_panel)
    output = noi.get_model_outputs(features, synthetic_panel)
    output = noi.add_rule_and_ensemble(output)
    return output


def _cohort_truth(df):
    """Ground truth: active indicators recur, dormant ones do not."""
    mask = df["indicator"].str.startswith(("active-", "dormant-"))
    subset = df[mask]
    y_true = subset["indicator"].str.startswith("active-").astype(int)
    return subset, y_true


class TestModelValidity:
    def test_probabilities_are_bounded(self, noi, scored):
        prob_cols = (
            [f"logistic_{h}" for h in noi.HORIZONS]
            + [f"gbt_{h}" for h in noi.HORIZONS]
            + [f"exp_{h}" for h in noi.HORIZONS]
            + [f"weibull_{h}" for h in noi.HORIZONS]
            + [f"ensemble_{h}d" for h in noi.HORIZONS]
        )
        for col in prob_cols:
            values = scored[col].astype(float).dropna()
            assert values.between(0, 1).all(), f"{col} out of [0, 1]"

    def test_exponential_model_is_monotone_in_horizon(self, noi, scored):
        # P(recurrence within h days) must not decrease as h grows
        for shorter, longer in zip(noi.HORIZONS, noi.HORIZONS[1:]):
            assert (
                scored[f"exp_{longer}"] >= scored[f"exp_{shorter}"] - 1e-12
            ).all()

    def test_weibull_model_is_monotone_in_horizon(self, noi, scored):
        for shorter, longer in zip(noi.HORIZONS, noi.HORIZONS[1:]):
            assert (
                scored[f"weibull_{longer}"] >= scored[f"weibull_{shorter}"] - 1e-12
            ).all()

    def test_single_class_target_returns_nan_not_crash(self, noi):
        X = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [0.0, 1.0, 0.0]})
        y = pd.Series([1, 1, 1])
        result = noi.train_predict_proba(noi.make_logistic, X, y)
        assert np.isnan(result).all()

    def test_ensemble_weights_are_a_convex_combination(self, noi):
        weights = noi.ENSEMBLE_WEIGHTS
        assert pytest.approx(sum(weights.values())) == 1.0
        assert all(w > 0 for w in weights.values())


class TestModelPerformance:
    """The models must separate recurring indicators from dormant ones."""

    AUC_FLOOR = 0.95  # cohort is cleanly separable; a sound model should ace it

    @pytest.mark.parametrize("model", ["logistic_7", "gbt_7", "weibull_7", "exp_7"])
    def test_individual_models_rank_active_above_dormant(self, scored, model):
        subset, y_true = _cohort_truth(scored)
        auc = roc_auc_score(y_true, subset[model].astype(float))
        assert auc >= self.AUC_FLOOR, f"{model} AUC {auc:.3f} below {self.AUC_FLOOR}"

    @pytest.mark.parametrize("horizon", [7, 14, 30, 45])
    def test_ensemble_ranks_active_above_dormant(self, scored, horizon):
        subset, y_true = _cohort_truth(scored)
        auc = roc_auc_score(y_true, subset[f"ensemble_{horizon}d"].astype(float))
        assert auc >= self.AUC_FLOOR, f"ensemble_{horizon}d AUC {auc:.3f}"

    def test_ensemble_recall_on_active_indicators(self, scored):
        # Every synthetic active indicator recurs within 7 days by construction;
        # the 7-day ensemble should assign each of them a substantial probability.
        active = scored[scored["indicator"].str.startswith("active-")]
        assert (active["ensemble_7d"].astype(float) >= 0.5).all()

    def test_dormant_indicators_score_low(self, scored):
        dormant = scored[scored["indicator"].str.startswith("dormant-")]
        assert (dormant["ensemble_7d"].astype(float) <= 0.3).all()


class TestAnalystOutput:
    def test_confidence_tiers_and_formatting(self, noi, scored):
        formatted = noi.add_confidence_and_format(scored.copy())
        for h in noi.HORIZONS:
            tiers = formatted[f"confidence_{h}d"].unique()
            allowed = {
                f"{h}-Day: Highly likely",
                f"{h}-Day: Possibly active",
                f"{h}-Day: Low confidence",
            }
            assert set(tiers) <= allowed
            assert formatted[f"ensemble_{h}d"].str.endswith("%").all()

    def test_production_table_schema(self, noi, scored):
        formatted = noi.add_confidence_and_format(scored.copy())
        production = noi.build_production_output(formatted)
        expected = (
            ["Indicator", "Observed Today", "Frequency (1d)", "Frequency (7d)", "Frequency (30d)"]
            + [c for h in noi.HORIZONS for c in (f"Probability: {h}-Day", f"Confidence: {h}-Day")]
        )
        assert list(production.columns) == expected
        assert len(production) == len(scored)
