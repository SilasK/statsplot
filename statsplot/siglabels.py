import matplotlib.pyplot as plt
import pandas as pd


def format_p_value(p, use_stars=False):
    """Format P values"""

    if not use_stars:
        return "P = {:.1g}".format(p).replace("0.", ".")
    else:
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        else:
            return "n.s."


from matplotlib.font_manager import FontProperties


def __plot_sig_label(
    x1, x2, y, text, ax=None, fontoffset=0, linewidth=1.5, font_size=None
):

    if font_size is None:
        font_size = "small"

    font = FontProperties(size=font_size)
    if ax is None:
        ax = plt.gca()

    ax.annotate(
        text,
        xy=((x1 + x2) / 2.0, y),
        textcoords="offset points",
        xytext=(0, fontoffset),
        horizontalalignment="center",
        verticalalignment="bottom",
        font_properties=font,
    )

    ax.plot(
        (x1, x2),
        (y, y),
        marker=3,
        markeredgewidth=linewidth,
        linewidth=linewidth,
        color="k",
    )


def __plot_sig_labels_hue(
    P_values,
    order,
    y0,
    deltay,
    ax,
    x_offset,
    show_ns=True,
    width=0.8,
    use_stars=True,
    **labelkws,
):

    assert type(P_values) == pd.Series, "P values should be a series"

    if not show_ns:
        P_values = P_values.loc[P_values < 0.05]

    P_values = P_values.apply(format_p_value, use_stars=use_stars)

    # start with y0
    y = y0

    for idx, text in P_values.iteritems():

        def calculate_hue_offset(group, order):
            return (order.index(group) - len(order) * 0.5 + 0.5) / len(order) * width

        x1 = calculate_hue_offset(idx.split("_vs_")[0], order) + x_offset
        x2 = calculate_hue_offset(idx.split("_vs_")[1], order) + x_offset

        __plot_sig_label(x1, x2, y, text, ax=ax, **labelkws)

        y += deltay


def ___plot_sig_labels_xaxis(
    P_values,
    order,
    y0,
    deltay,
    ax,
    show_ns=True,
    width=0.8,
    use_stars=True,
    **labelkws,
):

    assert type(P_values) == pd.Series, "P values should be a series"

    # remove non dignificant and format p values
    if not show_ns:
        P_values = P_values.loc[P_values < 0.05]

    if type(P_values) == pd.DataFrame:
        P_values = P_values.iloc[0]

    P_values = P_values.apply(format_p_value, use_stars=use_stars)

    y = y0
    for idx, text in P_values.iteritems():

        def calculate_x_offset(group, order):
            return order.index(group)

        x1 = calculate_x_offset(idx.split("_vs_")[0], order)
        x2 = calculate_x_offset(idx.split("_vs_")[1], order)

        __plot_sig_label(x1, x2, y, text, ax=ax, **labelkws)

        y += deltay


def plot_all_sig_labels(
    P_values,
    order,
    order_hue=None,
    show_ns=False,
    y0="auto",
    deltay="auto",
    ax=None,
    **kws,
):

    """"""

    # define y0 and deltay
    if ax is None:
        ax = plt.gca()

    Lim = ax.get_ylim()
    if deltay == "auto":
        deltay = (Lim[1] - Lim[0]) / 10
    if y0 == "auto":
        y0 = Lim[1] + (Lim[1] - Lim[0]) / 10

    order = list(order)

    # If order_hue is not None, then groups are seperated by the x axis
    if order_hue is None:
        ___plot_sig_labels_xaxis(
            P_values, order, ax=ax, deltay=deltay, y0=y0, show_ns=show_ns, **kws
        )
    # plot sig labels usign hue
    else:
        order_hue = list(order_hue)

        for cat, P in P_values.groupby(level=0):

            x_offset = order_hue.index(cat)

            __plot_sig_labels_hue(
                P_values.loc[cat],
                order,
                ax=ax,
                x_offset=x_offset,
                deltay=deltay,
                y0=y0,
                show_ns=show_ns,
                **kws,
            )
