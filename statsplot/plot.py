from logging import getLogger
from textwrap import dedent

logger = getLogger("__name__")

import matplotlib.pyplot as plt
import seaborn as sns
from numpy import unique, log10, abs
import pandas as pd

from .stats import calculate_stats
from .siglabels import plot_all_sig_labels


def _def_label_alignment(x, y):

    ha = "center"

    if abs(x) > abs(y):
        if x > 0:
            ha = "left"
        else:
            ha = "right"

    va = "center"

    if abs(y) > abs(x):
        if y > 0:
            va = "bottom"
        else:
            va = "top"

    return {"ha": ha, "va": va}


def annotate_points(
    *,
    data=None,
    x=None,
    y=None,
    labels=None,
    ax=None,
    max_labels=50,
    arrowprops=dict(arrowstyle="-", color="k", lw=0.5),
):

    if ax is None:
        ax = plt.gca()

    # parse data
    if data is None:
        assert (
            x is not None and y is not None
        ), "Either data or x and y should be provided"
    else:
        if x is None:
            assert data.shape[1] == 2, "Expect data n x 2"
            x = data.iloc[:, 0]
        elif type(x) == str:
            x = data.loc[:, x]

        if y is None:
            assert data.shape[1] == 2, "Expect data n x 2"
            y = data.iloc[:, 1]
        elif type(y) == str:
            y = data.loc[:, y]

        if labels is None:
            labels = data.index

    N_labels = len(labels)
    assert len(x) == N_labels, "x and labels should have the same length"
    assert len(y) == N_labels, "y and labels should have the same length"

    if N_labels > max_labels:

        logger.error(
            f"You want to label more than {max_labels} points."
            " This is would overcroud the plot. "
        )

    else:

        Texts = []
        for i in range(N_labels):

            Texts.append(
                plt.text(x[i], y[i], labels[i], **_def_label_alignment(x[i], y[i]))
            )

        try:
            from adjustText import adjust_text

            adjust_text(Texts, ax=ax, arrowprops=arrowprops)

        except ImportError:
            logger.warning(
                "Want to optimize label placement but adjustText is not installed."
                "This will inevitabely lead to overlapping labels."
                "You need to install it: `conda install -c conda-forge adjusttext` "
            )


def vulcanoplot(
    p_values,
    effect,
    hue=None,
    labels=None,
    threshold_p=0.05,
    figsize=(6, 6),
    label_points="auto",
    max_labels=5,
    **kws,
):

    f = plt.figure(figsize=figsize)

    assert effect.ndim == 1, "effect should be one dimensional"
    assert p_values.ndim == 1, "p_values should be one dimensional"

    logPvalues = -log10(p_values.astype(float))
    logPvalues.name = "$-\log(P)$"

    threshold = -log10(threshold_p)

    if hue is None:
        hue = pd.Series("Significant", index=logPvalues.index)

    ax = sns.scatterplot(
        y=logPvalues.loc[logPvalues > threshold],
        x=effect,
        hue=hue.loc[logPvalues > threshold],
    )

    ax = sns.scatterplot(
        y=logPvalues.loc[logPvalues <= threshold],
        x=effect,
        color="grey",
        marker=".",
        label="not significant",
    )

    if (label_points == "auto") or (label_points == True):

        label_data = pd.concat([effect, logPvalues], axis=1)
        label_data.columns = ["effect", "logP"]

        if labels is not None:
            label_data["Labels"] = labels

        if label_points == "auto":
            # select only significant points
            label_data = label_data.query("logP > @threshold")

            max_labels = min(max_labels, label_data.shape[0])

        if max_labels > 0:

            # calculate radius to detect outermost labels based on effect and logP
            effect_range = abs(label_data.effect.max() - label_data.effect.min())
            logP_range = label_data.logP.max() - label_data.logP.min()
            # subtract minimum log P value this changes the radius if only significat points are selected
            min_log_p = label_data.logP.min()

            label_data.eval(
                "radius = (abs(effect)/ @effect_range)**2 + ((logP - @min_log_p )/@logP_range )**2 ",
                inplace=True,
            )

            label_data = label_data.sort_values("radius", ascending=False)

            label_data = label_data.iloc[:max_labels]

            annotate_points(
                data=label_data.iloc[:, :2],
                labels=label_data.Labels,
                max_labels=max_labels,
            )

    # legend
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=10)
    # equalize axis
    ax_lim = abs(ax.get_xlim()).max()
    ax.set_xlim([-ax_lim, ax_lim])

    if "_vs_" in effect.name:
        g1, g2 = effect.name.split("_vs_")

        ax.annotate(g1, (ax_lim * 0.9, 0), ha="right")
        ax.annotate(g2, (-ax_lim * 0.9, 0), ha="left")

# TODO: handle unaligned input.

def statsplot(
    variable,
    test_variable,
    data=None,
    order_test=None,
    grouping_variable=None,
    order_grouping=None,
    show_dots=True,
    box_params=None,
    swarm_params=None,
    labelkws=None,
    stats_kws=None,
    palette=None,
    p_values=None,
    test="ttest_ind",
    show_not_significant=False,
    ax=None,
):
    """Main function for plotting statistical tests."""
    if ax is None:
        ax = plt.subplot(111)

    if type(variable) == str:
        assert data is not None, "If variable is a string, data must be provided"

        variable = data[variable]

    if type(test_variable) == str:
        assert data is not None, "If test_variable is a string, data must be provided"
        test_variable = data[test_variable]

    params = dict(y=variable, ax=ax)

    if order_test is None:
        order_test = unique(test_variable)

    # use subgrouping
    if grouping_variable is None:
        params.update(dict(x=test_variable, order=order_test))
    else:

        if type(grouping_variable) == str:
            assert (
                data is not None
            ), "If grouping_variable is a string, data must be provided"
            grouping_variable = data[grouping_variable]

        if order_grouping is None:
            order_grouping = unique(grouping_variable)

        params.update(
            dict(
                x=grouping_variable,
                order=order_grouping,
                hue=test_variable,
                hue_order=order_test,
            )
        )

    if box_params is None:
        box_params = {}

    if swarm_params is None:
        swarm_params = {}

    sns.boxplot(palette=palette, **params, **box_params)

    if show_dots:
        legend = ax.get_legend_handles_labels()

        sns.swarmplot(**params, color="k", dodge=True, **swarm_params)

        # add old variable, as dots have unified colors
        if grouping_variable is not None:
            ax.legend(
                *legend, bbox_to_anchor=(1, 1), title=ax.legend_.get_title().get_text()
            )

    # Statistics
    if p_values is None:

        if stats_kws is None:
            stats_kws = dict()

        p_values = calculate_stats(
            variable,
            test_variable,
            grouping_variable=grouping_variable,
            test=test,
            **stats_kws,
        ).Pvalue.iloc[0]

    if labelkws is None:
        labelkws = dict(deltay="auto")
        if show_not_significant:
            labelkws.update(use_stars=False)

    plot_all_sig_labels(
        p_values,
        order_test,
        order_grouping,
        show_ns=show_not_significant,
        ax=ax,
        **labelkws,
    )

    return ax, p_values


statsplot.__doc__ = dedent(
    """\
    Plot Boxplot with statistical significance.

    Parameters  
    ----------
    variable : str or pandas.Series
        The variable to be tested.
    test_variable : str or pandas.Series
        The variable to be tested against.
    data : pandas.DataFrame
        The dataframe containing the variable, test_variable and grouping_variable.
    order_test : list           
        The order of the test_variable.
    grouping_variable : str or pandas.Series
        The variable to be used for grouping.
    order_grouping : list
        The order of the grouping_variable.
    show_dots : bool
        If True, show dots on top of boxplot.
    box_params : dict
        Parameters for the boxplot.
    swarm_params : dict
        Parameters for the swarmplot.
    labelkws : dict
        Parameters for the labels.
    stats_kws : dict
        Parameters for the statistical test.
    palette : list
        The color palette.
    p_values : pandas.Series
        The p-values of the statistical test.
    test : str
        The statistical test.
    show_not_significant : bool
        If True, show not significant labels.
    ax : matplotlib.axes.Axes
        The axis to plot on.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the plot.
    p_values : pandas.Series
        The p-values of the statistical test.

    Examples
    --------


    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> import seaborn as sns
        >>> import statsplot as stp
        >>> iris = sns.load_dataset("iris")
        >>> ax,stats = stp.statsplot(data=iris, variable="sepal_length", test_variable="species")
        >>> print(stats)
        versicolor_vs_setosa       8.985235e-18
        virginica_vs_setosa        6.892546e-28
        virginica_vs_versicolor    1.724856e-07
        Name: sepal_length, dtype: float64
        >>> plt.show()



    """
)
