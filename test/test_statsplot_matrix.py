from __future__ import annotations

import pandas as pd
import pytest
import seaborn as sns
from matplotlib.axes import Axes

from statsplot import statsplot


@pytest.mark.parametrize(
    "case",
    [
        {
            "name": "ungrouped_no_dots",
            "grouped": False,
            "show_dots": False,
            "swarm_params": None,
        },
        {
            "name": "ungrouped_with_dots",
            "grouped": False,
            "show_dots": True,
            "swarm_params": None,
        },
        {
            "name": "grouped_no_dots",
            "grouped": True,
            "show_dots": False,
            "swarm_params": None,
        },
        {
            "name": "grouped_with_dots_default",
            "grouped": True,
            "show_dots": True,
            "swarm_params": None,
        },
        {
            "name": "grouped_with_dots_dodge_only",
            "grouped": True,
            "show_dots": True,
            "swarm_params": {"dodge": True},
        },
        {
            "name": "grouped_with_dots_partial_swarm",
            "grouped": True,
            "show_dots": True,
            "swarm_params": {"size": 2},
        },
        {
            "name": "grouped_with_dots_override_dodge_false",
            "grouped": True,
            "show_dots": True,
            "swarm_params": {"dodge": False},
        },
    ],
    ids=lambda c: c["name"],
)
def test_statsplot_behavior_matrix(
    case,
    synthetic_data: pd.DataFrame,
    p_values_ungrouped: pd.Series,
    p_values_grouped: pd.Series,
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, dict] = {}

    original_boxplot = sns.boxplot
    original_swarmplot = sns.swarmplot

    def boxplot_spy(*args, **kwargs):
        captured["box"] = dict(kwargs)
        return original_boxplot(*args, **kwargs)

    def swarmplot_spy(*args, **kwargs):
        captured["swarm"] = dict(kwargs)
        return original_swarmplot(*args, **kwargs)

    monkeypatch.setattr(sns, "boxplot", boxplot_spy)
    monkeypatch.setattr(sns, "swarmplot", swarmplot_spy)

    grouped = case["grouped"]
    show_dots = case["show_dots"]

    p_values = p_values_grouped if grouped else p_values_ungrouped
    grouping_variable = "group_bin" if grouped else None
    order_grouping = ["G1", "G2"] if grouped else None

    ax, returned_p_values = statsplot(
        variable="value",
        test_variable="test_group",
        data=synthetic_data,
        order_test=["A", "B"],
        grouping_variable=grouping_variable,
        order_grouping=order_grouping,
        show_dots=show_dots,
        swarm_params=case["swarm_params"],
        p_values=p_values,
    )

    # Smoke checks: all matrix rows should execute and return axis + p-values.
    assert isinstance(ax, Axes)
    assert isinstance(returned_p_values, pd.Series)

    box_kwargs = captured["box"]
    if show_dots:
        assert box_kwargs["showfliers"] is False
        assert box_kwargs["fliersize"] == 0
    else:
        assert "showfliers" not in box_kwargs
        assert "fliersize" not in box_kwargs

    if not show_dots:
        assert "swarm" not in captured
        return

    swarm_kwargs = captured["swarm"]

    if not grouped:
        # No-grouping path should not require grouped hue/dodge defaults.
        assert "dodge" not in swarm_kwargs
        assert "hue" not in swarm_kwargs
        assert "hue_order" not in swarm_kwargs
        assert swarm_kwargs.get("color") == "k"
        return

    # Grouped + dots path should use dodge defaults unless explicitly overridden.
    if case["swarm_params"] == {"dodge": False}:
        assert swarm_kwargs["dodge"] is False
    else:
        assert swarm_kwargs["dodge"] is True

    # Grouped path should preserve hue defaults even with partial swarm kwargs.
    assert "hue" in swarm_kwargs
    assert set(pd.unique(swarm_kwargs["hue"])) == {"A", "B"}
    assert swarm_kwargs["hue_order"] == ["A", "B"]
    assert swarm_kwargs["palette"] == "dark:k"

    if case["swarm_params"] == {"size": 2}:
        assert swarm_kwargs["size"] == 2
