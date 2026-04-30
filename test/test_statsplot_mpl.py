from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images

from statsplot import statsplot


BASELINE_DIR = Path(__file__).parent / "baseline_images"


def _assert_matches_baseline(fig, baseline_name: str, tolerance: float = 2.0):
    actual_dir = Path(__file__).parent / "_actual"
    actual_dir.mkdir(exist_ok=True)

    actual_path = actual_dir / baseline_name
    fig.savefig(actual_path, dpi=100, bbox_inches="tight")

    expected_path = BASELINE_DIR / baseline_name
    assert expected_path.exists(), f"Missing baseline image: {expected_path}"

    result = compare_images(str(expected_path), str(actual_path), tol=tolerance)
    assert result is None, result


def test_mpl_grouped_dots_default(synthetic_data, p_values_grouped):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
    statsplot(
        variable="value",
        test_variable="test_group",
        data=synthetic_data,
        order_test=["A", "B"],
        grouping_variable="group_bin",
        order_grouping=["G1", "G2"],
        show_dots=True,
        p_values=p_values_grouped,
        ax=ax,
    )
    _assert_matches_baseline(fig, "grouped_dots_default.png")
    plt.close(fig)


def test_mpl_grouped_dots_partial_swarm_params(synthetic_data, p_values_grouped):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
    statsplot(
        variable="value",
        test_variable="test_group",
        data=synthetic_data,
        order_test=["A", "B"],
        grouping_variable="group_bin",
        order_grouping=["G1", "G2"],
        show_dots=True,
        swarm_params={"size": 2},
        p_values=p_values_grouped,
        ax=ax,
    )
    _assert_matches_baseline(fig, "grouped_dots_partial_swarm.png")
    plt.close(fig)


def test_mpl_ungrouped_dots(synthetic_data, p_values_ungrouped):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
    statsplot(
        variable="value",
        test_variable="test_group",
        data=synthetic_data,
        order_test=["A", "B"],
        show_dots=True,
        p_values=p_values_ungrouped,
        ax=ax,
    )
    _assert_matches_baseline(fig, "ungrouped_dots.png")
    plt.close(fig)
