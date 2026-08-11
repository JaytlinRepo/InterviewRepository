"""Shared fixtures for the NOI model test suite.

The model logic lives in the research notebook. Rather than duplicating it,
these tests load the notebook's function definitions and constants directly,
so the suite always validates the exact code an interviewer reads.
"""
import ast
import json
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

NOTEBOOK = (
    Path(__file__).resolve().parent.parent
    / "observationEventForecasting"
    / "NextObservedIndicatorV3.0.ipynb"
)


def _load_notebook_namespace():
    """Execute only the imports, constants, and function defs from the notebook.

    Data-loading and display statements are skipped, so nothing here touches
    the (redacted) data paths.
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
        code = compile(ast.Module(body=keep, type_ignores=[]), str(NOTEBOOK), "exec")
        exec(code, module.__dict__)
    return module


@pytest.fixture(scope="session")
def noi():
    """Namespace holding the notebook's functions and constants."""
    return _load_notebook_namespace()


def make_observations(schedules, start="2025-01-01", end="2025-04-10", opdiv="TEST"):
    """Build a raw observation frame from {indicator: [day offsets seen]}."""
    dates = pd.date_range(start, end, freq="D")
    rows = []
    for indicator, offsets in schedules.items():
        for offset in offsets:
            rows.append(
                {
                    "indicator": indicator,
                    "API_UserName": "user-1",
                    "date": dates[offset],
                    "OpDiv": opdiv,
                    "observations": 5,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def synthetic_panel(noi):
    """Dense panel with clearly separable active vs. dormant indicators.

    Active indicators keep appearing through the end of the window; dormant
    ones went quiet long ago. A seeded random cohort adds realistic noise.
    """
    rng = np.random.default_rng(7)
    n_days = 100
    schedules = {}
    # Irregular cadences and per-indicator rates keep the survival model's
    # duration distribution non-degenerate (constant gaps break AFT fitting).
    for i in range(10):  # active: frequent, irregular, and seen recently
        rate = rng.uniform(0.25, 0.55)
        days = [d for d in range(n_days) if rng.random() < rate]
        days.append(n_days - 1 - int(rng.integers(0, 3)))  # guarantee recent hit
        schedules[f"active-{i}"] = sorted(set(days))
    for i in range(10):  # dormant: sporadic activity only in the first 30 days
        rate = rng.uniform(0.1, 0.3)
        days = [d for d in range(30) if rng.random() < rate] or [int(rng.integers(0, 30))]
        schedules[f"dormant-{i}"] = sorted(set(days))
    for i in range(10):  # noise: sparse random activity across the window
        schedules[f"noise-{i}"] = sorted(
            rng.choice(np.arange(0, n_days), size=int(rng.integers(3, 7)), replace=False).tolist()
        )

    raw = make_observations(schedules, start="2025-01-01", end="2025-04-10")
    panel = noi.build_dense_panel(raw)
    # Pin the panel to the synthetic window: build_dense_panel extends to the
    # real "today", which would make every synthetic indicator look dormant.
    return panel[panel["date"] <= pd.Timestamp("2025-04-10")].copy()
