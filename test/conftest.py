from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest


# CI-safe, headless backend for deterministic rendering.
matplotlib.use("Agg")


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    rng = np.random.default_rng(7)

    n = 20
    rows = []
    for group_bin, offset in (("G1", 0.0), ("G2", 0.35)):
        for test_group, mean in (("A", 0.0 + offset), ("B", 0.9 + offset)):
            values = rng.normal(loc=mean, scale=0.25, size=n)
            rows.extend(
                {
                    "value": float(v),
                    "test_group": test_group,
                    "group_bin": group_bin,
                }
                for v in values
            )

    data = pd.DataFrame(rows)

    # Inject explicit outliers so showfliers behavior is testable.
    outliers = pd.DataFrame(
        [
            {"value": 4.2, "test_group": "A", "group_bin": "G1"},
            {"value": -3.4, "test_group": "B", "group_bin": "G1"},
            {"value": 4.8, "test_group": "A", "group_bin": "G2"},
            {"value": -3.9, "test_group": "B", "group_bin": "G2"},
        ]
    )
    return pd.concat([data, outliers], ignore_index=True)


@pytest.fixture
def p_values_ungrouped() -> pd.Series:
    return pd.Series({"B_vs_A": 0.01}, name="value")


@pytest.fixture
def p_values_grouped() -> pd.Series:
    idx = pd.MultiIndex.from_tuples(
        [("G1", "B_vs_A"), ("G2", "B_vs_A")],
        names=["group_bin", "comparison"],
    )
    return pd.Series([0.01, 0.02], index=idx, name="value")
