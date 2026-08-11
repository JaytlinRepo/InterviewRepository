"""Correctness tests for dense-panel construction and feature extraction."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_observations


def _features_for(noi, seen_series):
    """Run extract_time_series_features on an explicit 0/1 daily series."""
    group = pd.DataFrame({"seen": seen_series})
    return noi.extract_time_series_features(group)


class TestDensePanel:
    def test_grid_is_complete(self, noi):
        raw = make_observations({"a": [0, 5], "b": [3]}, start="2025-01-01", end="2025-01-10")
        panel = noi.build_dense_panel(raw)
        n_days = (panel["date"].max() - panel["date"].min()).days + 1
        assert len(panel) == 2 * n_days  # 1 source x n_days x 2 indicators

    def test_missing_days_are_zero_filled(self, noi):
        raw = make_observations({"a": [0]}, start="2025-01-01", end="2025-01-10")
        panel = noi.build_dense_panel(raw)
        silent = panel[panel["indicator"] == "a"].iloc[1:]
        assert (silent["observations"] == 0).all()
        assert (silent["seen"] == 0).all()

    def test_seen_is_binary_indicator_of_observations(self, noi):
        raw = make_observations({"a": [0, 2], "b": [1]}, start="2025-01-01", end="2025-01-05")
        panel = noi.build_dense_panel(raw)
        assert set(panel["seen"].unique()) <= {0, 1}
        assert (panel["seen"] == (panel["observations"] > 0).astype(int)).all()

    def test_weekend_flag_matches_calendar(self, noi):
        raw = make_observations({"a": [0]}, start="2025-01-01", end="2025-01-14")
        panel = noi.build_dense_panel(raw)
        assert (panel["is_weekend"] == panel["date"].dt.dayofweek.isin([5, 6])).all()


class TestFeatureExtraction:
    def test_never_seen_indicator_gets_sentinel_values(self, noi):
        feats = _features_for(noi, np.zeros(60, dtype=int))
        assert feats["last_seen"] == 60
        assert feats["avg_gap"] == 60
        assert all(feats[f"freq_{w}"] == 0 for w in noi.HORIZONS)
        assert all(feats[f"label_{w}"] == 0 for w in noi.LABEL_HORIZONS)

    def test_last_seen_counts_days_since_most_recent_observation(self, noi):
        series = np.zeros(50, dtype=int)
        series[44] = 1  # seen 5 days before the end of the window
        feats = _features_for(noi, series)
        assert feats["last_seen"] == 5

    def test_windowed_frequencies(self, noi):
        series = np.zeros(60, dtype=int)
        series[[59, 57, 54, 40, 20]] = 1
        feats = _features_for(noi, series)
        assert feats["freq_1"] == 1     # day 59
        assert feats["freq_7"] == 3     # days 59, 57, 54
        assert feats["freq_30"] == 4    # + day 40
        assert feats["freq_45"] == 5    # + day 20

    def test_labels_flag_any_activity_in_window(self, noi):
        series = np.zeros(60, dtype=int)
        series[49] = 1  # 10 days before the end: outside 7d, inside 14/30/45d
        feats = _features_for(noi, series)
        assert feats["label_7"] == 0
        assert feats["label_14"] == 1
        assert feats["label_30"] == 1
        assert feats["label_45"] == 1

    def test_avg_gap_of_periodic_series(self, noi):
        series = np.zeros(60, dtype=int)
        series[::3] = 1  # every 3 days
        feats = _features_for(noi, series)
        assert feats["avg_gap"] == pytest.approx(3.0)

    def test_burstiness_sign_separates_periodic_from_bursty(self, noi):
        periodic = np.zeros(60, dtype=int)
        periodic[::4] = 1  # constant gaps -> burstiness = -1
        assert _features_for(noi, periodic)["burstiness"] == pytest.approx(-1.0)

        bursty = np.zeros(60, dtype=int)
        bursty[[0, 1, 2, 3, 59]] = 1  # tight cluster then a long silence
        assert _features_for(noi, bursty)["burstiness"] > 0

    def test_build_features_emits_one_row_per_indicator(self, noi, synthetic_panel):
        features = noi.build_features(synthetic_panel)
        assert len(features) == synthetic_panel["indicator"].nunique()
        assert set(noi.FEATURE_COLS).issubset(features.columns)
