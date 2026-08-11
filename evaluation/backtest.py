"""Walk-forward backtest for the NOI forecasting pipeline.

Evaluates the production scoring procedure the honest way:

    1. Freeze the panel at a cutoff date T (the model sees nothing after T).
    2. Run the exact pipeline from the modeling notebook on that snapshot.
    3. Grade every forecast against what actually happened in (T, T+h] —
       forward-looking labels that share no window with the features.
    4. Roll T forward and repeat; aggregate metrics across cutoffs.

This complements the trailing-window labels used inside the daily pipeline:
those fit each morning's ranking snapshot, while this harness measures true
forward performance — each question answered with the right instrument.

Also reports a naive recency baseline ("predict recurrence iff seen in the
last h days") so the ensemble's lift is visible, plus a calibration table.

Usage:
    python evaluation/backtest.py                 # synthetic cohort (default)
    python evaluation/backtest.py --data-dir DIR  # real daily extracts
"""
import argparse
import ast
import json
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / "observationEventForecasting" / "NextObservedIndicatorV3.0.ipynb"
CUTOFF_STRIDE_DAYS = 7
N_CUTOFFS = 3
MODELS = ["logistic", "gbt", "exp", "weibull"]


# --------------------------------------------------------------------------- notebook loader
def load_noi_namespace():
    """Load the notebook's functions/constants (single source of truth).

    Executes only imports, UPPERCASE constants, and function definitions —
    never the data-loading or display statements.
    """
    nb = json.loads(NOTEBOOK.read_text())
    module = types.ModuleType("noi")
    module.display = lambda *args, **kwargs: None
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        keep = [
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
            or (
                isinstance(node, ast.Assign)
                and all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets)
            )
        ]
        exec(compile(ast.Module(body=keep, type_ignores=[]), str(NOTEBOOK), "exec"), module.__dict__)
    return module


# --------------------------------------------------------------------------- data
def make_synthetic_observations(n_days=160, seed=11):
    """Seeded synthetic cohort mirroring real behavioral archetypes.

    - active:  frequent, irregular sightings through the whole window
    - dormant: activity confined to the first ~40 days, silent after
    - waning:  cadence that slows over time (hard cases near the boundary)
    - noise:   sparse, unpredictable sightings
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    schedules = {}
    for i in range(12):
        rate = rng.uniform(0.2, 0.5)
        schedules[f"active-{i}"] = [d for d in range(n_days) if rng.random() < rate]
    for i in range(12):
        rate = rng.uniform(0.15, 0.35)
        schedules[f"dormant-{i}"] = [d for d in range(40) if rng.random() < rate]
    for i in range(8):
        schedules[f"waning-{i}"] = [
            d for d in range(n_days) if rng.random() < 0.4 * np.exp(-d / rng.uniform(40, 80))
        ]
    for i in range(8):
        schedules[f"noise-{i}"] = sorted(
            rng.choice(np.arange(n_days), size=int(rng.integers(3, 10)), replace=False).tolist()
        )

    rows = [
        {"indicator": ind, "API_UserName": "user-1", "date": dates[d],
         "OpDiv": "TEST", "observations": int(rng.integers(1, 40))}
        for ind, days in schedules.items()
        for d in sorted(set(days))
    ]
    return pd.DataFrame(rows)


def load_real_observations(data_dir):
    """Load real daily extracts (same schema as the modeling notebook)."""
    frames = [pd.read_csv(p) for p in sorted(Path(data_dir).glob("*.csv"))]
    src = pd.concat(frames, ignore_index=True)
    src["indicator"] = src["indicator"].astype(str).str.split(" ", expand=True)[0].str.strip()
    src["OpDiv"] = src["OpDiv"].astype(str).str.strip()
    return (
        src.drop(columns=["curr_date", "indicator_key"], errors="ignore")
           .rename(columns={"obs_date": "date"})
           .assign(date=lambda d: pd.to_datetime(d["date"]))
    )


# --------------------------------------------------------------------------- backtest core
def score_snapshot(noi, panel, cutoff):
    """Run the notebook pipeline on data up to `cutoff` only."""
    snapshot = panel[panel["date"] <= cutoff].copy()
    features = noi.build_features(snapshot)
    output = noi.get_model_outputs(features, snapshot)
    return noi.add_rule_and_ensemble(output)


def forward_labels(observations, cutoff, horizons):
    """Ground truth: was each indicator actually observed in (cutoff, cutoff+h]?"""
    labels = {}
    for h in horizons:
        seen = observations[
            (observations["date"] > cutoff)
            & (observations["date"] <= cutoff + pd.Timedelta(days=h))
        ]["indicator"].unique()
        labels[h] = set(seen)
    return labels


def run_backtest(noi, observations, cutoffs):
    """Score at each cutoff, grade against forward outcomes, return records."""
    panel = noi.build_dense_panel(observations)
    panel = panel[panel["date"] <= observations["date"].max()]

    records = []
    for cutoff in cutoffs:
        try:
            scored = score_snapshot(noi, panel, cutoff)
        except Exception as exc:  # e.g. survival-model convergence on a degenerate snapshot
            print(f"  [skip] cutoff {cutoff:%Y-%m-%d}: {type(exc).__name__}: {exc}")
            continue
        truth = forward_labels(observations, cutoff, noi.HORIZONS)
        for _, row in scored.iterrows():
            rec = {"cutoff": cutoff, "indicator": row["indicator"], "last_seen": row["last_seen"]}
            for h in noi.HORIZONS:
                rec[f"y_true_{h}"] = int(row["indicator"] in truth[h])
                rec[f"ensemble_{h}"] = float(row[f"ensemble_{h}d"])
                for m in MODELS:
                    rec[f"{m}_{h}"] = float(row[f"{m}_{h}"])
            records.append(rec)
        print(f"  cutoff {cutoff:%Y-%m-%d}: scored {len(scored)} indicators")
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- reporting
def safe_auc(y_true, y_score):
    mask = ~np.isnan(y_score)
    if y_true[mask].nunique() < 2:
        return np.nan
    return roc_auc_score(y_true[mask], y_score[mask])


def report(results, horizons):
    print("\n" + "=" * 74)
    print("WALK-FORWARD BACKTEST — forward-looking labels, aggregated over cutoffs")
    print("=" * 74)

    summary = []
    for h in horizons:
        y = results[f"y_true_{h}"]
        naive = (results["last_seen"] <= h - 1).astype(int)  # recency rule baseline
        row = {
            "horizon": f"{h}d",
            "base_rate": y.mean(),
            "naive_precision": precision_score(y, naive, zero_division=0),
            "naive_recall": recall_score(y, naive, zero_division=0),
        }
        for m in MODELS:
            row[f"auc_{m}"] = safe_auc(y, results[f"{m}_{h}"])
        row["auc_ensemble"] = safe_auc(y, results[f"ensemble_{h}"])
        row["ap_ensemble"] = average_precision_score(y, results[f"ensemble_{h}"])
        row["brier_ensemble"] = brier_score_loss(y, results[f"ensemble_{h}"].clip(0, 1))
        summary.append(row)

    table = pd.DataFrame(summary).set_index("horizon")
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print("\nPer-horizon metrics (AUC per model; AP/Brier for the ensemble):\n")
        print(table.to_string())

    # Calibration of the 7-day ensemble: predicted probability vs observed rate
    print("\nCalibration — 7-day ensemble (predicted bin vs. observed recurrence):\n")
    bins = pd.cut(results["ensemble_7"], bins=[0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    calib = results.groupby(bins, observed=True).agg(
        mean_predicted=("ensemble_7", "mean"),
        observed_rate=("y_true_7", "mean"),
        n=("y_true_7", "size"),
    )
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(calib.to_string())

    best_single = table[[f"auc_{m}" for m in MODELS]].max(axis=1)
    lift = (table["auc_ensemble"] - best_single).mean()
    print(f"\nEnsemble AUC vs. best single model, mean across horizons: {lift:+.3f}")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", help="Directory of real daily extract CSVs "
                                           "(defaults to a seeded synthetic cohort)")
    args = parser.parse_args()

    noi = load_noi_namespace()
    if args.data_dir:
        observations = load_real_observations(args.data_dir)
        print(f"Loaded {len(observations):,} rows from {args.data_dir}")
    else:
        observations = make_synthetic_observations()
        print(f"Using seeded synthetic cohort: {observations['indicator'].nunique()} "
              f"indicators over {observations['date'].nunique()} active days")

    last = observations["date"].max()
    max_h = max(noi.HORIZONS)
    cutoffs = [
        last - pd.Timedelta(days=max_h + i * CUTOFF_STRIDE_DAYS)
        for i in reversed(range(N_CUTOFFS))
    ]
    print(f"Cutoffs: {[f'{c:%Y-%m-%d}' for c in cutoffs]} "
          f"(each leaves {max_h}+ days of forward outcomes)\n")

    results = run_backtest(noi, observations, cutoffs)
    if results.empty:
        raise SystemExit("No cutoffs produced results.")
    report(results, noi.HORIZONS)


if __name__ == "__main__":
    main()
