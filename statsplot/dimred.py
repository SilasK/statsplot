from logging import getLogger

logger = getLogger("__name__")

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import warnings
from .plot import annotate_points, _def_label_alignment
import seaborn as sns


from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from scipy.stats import chi2


def confidence_ellipse(x, y, ax, ci=0.95, color="red", facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    if ax is None:
        ax = plt.gca()

    if len(x) < 4:
        raise Exception("need more than 3 data points")

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        edgecolor=color,
        **kwargs,
    )

    s = chi2.ppf(ci, 2)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0] * s)
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1] * s)
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_confidence_ellipses(
    x,
    y,
    groups,
    order=None,
    colors=None,
    confidence_interval=0.95,
    facecolor="none",
    ax=None,
    **kwargs,
):

    x = np.array(x)
    y = np.array(y)

    if ax is None:
        ax = plt.subplot(111)

    if order is None:
        order = np.unique(groups)

    if colors is None:
        colors = sns.color_palette(n_colors=len(order))

    if kwargs is None:
        kwargs = {}

    for n, g in enumerate(order):
        confidence_ellipse(
            x[groups == g],
            y=y[groups == g],
            ax=ax,
            color=colors[n],
            ci=confidence_interval,
            facecolor=facecolor,
            **kwargs,
        )

    return ax


class DimRed:
    def __init__(
        self, data, method=PCA, transformation=None, n_components=None, **kargs
    ):

        if n_components is None:
            n_components = data.shape[0]

        if data.shape[0] > data.shape[1]:
            print(
                "you don't need to reduce dimensionality or your dataset is transposed."
            )

        self.decomposition = method(n_components=n_components, **kargs)

        self.rawdata = data

        # self.variable_names = data.columns

        # self.sample_names = data.index

        if transformation is None:
            self.data_ = self.rawdata

        else:

            self.data_ = data.applymap(transformation)

        Xt = self.decomposition.fit_transform(self.data_)

        self.transformed_data = pd.DataFrame(
            Xt[:, : (n_components + 1)],
            index=data.index,
            columns=np.arange(n_components) + 1,
        )

        name_components = ["components_"]

        for name in name_components:
            if hasattr(self.decomposition, name):
                self.components = pd.DataFrame(
                    getattr(self.decomposition, name),
                    index=np.arange(n_components) + 1,
                    columns=data.columns,
                )

        if not hasattr(self, "components"):
            warnings.warn(
                "Couldn't define components, wil not be able to plot loadings"
            )

    def set_axes_labels_(self, ax, components):
        if hasattr(self.decomposition, "explained_variance_ratio_"):

            ax.set_xlabel(
                "PC {} [{:.1f} %]".format(
                    components[0],
                    self.decomposition.explained_variance_ratio_[components[0] - 1]
                    * 100,
                )
            )
            ax.set_ylabel(
                "PC {} [{:.1f} %]".format(
                    components[1],
                    self.decomposition.explained_variance_ratio_[components[1] - 1]
                    * 100,
                )
            )

        else:

            ax.set_xlabel("Component {} ".format(components[0]))
            ax.set_ylabel("Component {} ".format(components[1]))

    def plot_explained_variance_ratio(self, n_components=25, **kwargs):

        explained_variance_ratio = self.decomposition.explained_variance_ratio_
        n = min(n_components, len(explained_variance_ratio))

        plt.bar(np.arange(n), explained_variance_ratio[:n], **kwargs)

        ax = plt.gca()
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance Ratio")

        return ax

    def plot_components(
        self,
        components=(1, 2),
        ax=None,
        groups=None,
        plot_ellipse=False,
        label_points=False,
        confidence_interval=0.95,
        order_groups=None,
        colors=None,
        **scatter_args,
    ):

        components = list(components)
        assert len(components) == 2, "expect two components"

        if ax is None:
            ax = plt.subplot(111)

        if (groups is not None) and (order_groups is None):
            order_groups = np.unique(groups)

        x, y = (
            self.transformed_data[components[0]],
            self.transformed_data[components[1]],
        )

        overwritten_seaborn_kargs = {
            "hue": "groups",
            "hue_order": "order_groups",
            "palette": "colros",
        }

        for k in overwritten_seaborn_kargs:
            if k in scatter_args:
                raise ValueError(
                    f"You provided `{k}` as keyword. However `{k}` is overwritten by the `{overwritten_seaborn_kargs[k]}` argument."
                )

        sns.scatterplot(
            x=x,
            y=y,
            ax=ax,
            hue=groups,
            hue_order=order_groups,
            palette=colors,
            **scatter_args,
        )

        ax.axis("equal")
        self.set_axes_labels_(ax, components)
        if label_points:
            annotate_points(data=self.transformed_data[components], ax=ax)

        if plot_ellipse:
            if groups is None:
                raise Exception("`groups`` is required for plotting confidence ellipse")

            plot_confidence_ellipses(
                x,
                y,
                groups,
                order=order_groups,
                colors=colors,
                confidence_interval=confidence_interval,
                ax=ax,
            )

        return ax

    def plot_loadings(self, components=(1, 2), ax=None, **scatter_args):

        if ax is None:
            ax = plt.subplot(111)

        components = list(components)
        assert len(components) == 2, "expect two components"

        sns.scatterplot(
            x=self.components.loc[components[0]],
            y=self.components.loc[components[1]],
            ax=ax,
            **scatter_args,
        )
        ax.axis("equal")
        self.set_axes_labels_(ax, components)

        return ax

    def _detect_which_arrows_to_vizualize(self, loadings, n_arrows=None):

        assert loadings.shape[0] == 2

        radius = np.sqrt(sum(loadings.values**2))

        radius = pd.Series(radius, self.components.columns).sort_values(ascending=False)

        if n_arrows is None:

            try:
                from kneed import KneeLocator

                kneedle = KneeLocator(
                    np.arange(radius.shape[0]),
                    radius.values,
                    S=1.0,
                    curve="convex",
                    direction="decreasing",
                )

                n_arrows = kneedle.knee

                if n_arrows > 8:
                    print(
                        f"automatic selection selected {n_arrows} to visualize, which is probably to much. I select only 8"
                    )
                    n_arrows = 8

            except ImportError as e:

                print(
                    "Optional dependency 'kneed' is not installed. I cannot guess optimal number of components (arrows) to visualize."
                    "Specify them or install the package. Choose 4"
                )

                n_arrows = 4

        return list(radius.index[:n_arrows])

    def plot_biplot(
        self, components=[1, 2], n_arrows=None, scale_factor=None, labels=None, **kws
    ):

        ax = self.plot_components(**kws)

        if scale_factor is None:
            scale_factor = max(
                self.transformed_data[components].max()
                - self.transformed_data[components].min()
            )

        loadings = self.components.loc[components]

        interesting_components = self._detect_which_arrows_to_vizualize(
            loadings, n_arrows=n_arrows
        )

        Texts = []
        for c in interesting_components:

            x, y = loadings[c] * scale_factor

            ax.arrow(0, 0, x, y, alpha=0.5, width=0.3, color="k", linewidth=0)

            if labels is None:
                label = c
            else:
                label = labels[c]

            Texts.append(
                ax.text(
                    x * 1.3, y * 1.3, label, color="k", **_def_label_alignment(x, y)
                )
            )

        try:
            from adjustText import adjust_text

            adjust_text(Texts, x=[0], y=[0], ax=ax)

        except ImportError:
            logger.warning(
                "Want to optimize label placement but adjustText is not installed."
                "This will inevitabely lead to overlapping labels."
                "You need to install it: `conda install -c conda-forge adjusttext` "
            )

        return ax


## Interactive
import warnings

try:
    import altair as alt

    def altair_plot2D(data, variables=None, **kws):

        if variables is None:
            variables = data.columns[:2]

        plot_data = data[variables].copy()

        if plot_data.shape[0] > 5000:

            warnings.warn(
                "The number of rows in your dataset is greater than the maximum allowed (5000). I subsample. For information on how to plot larger datasets in Altair, see the documentation"
            )

            plot_data = plot_data.loc[
                plot_data.abs().max(1).sort_values(ascending=False).index[:4998]
            ]

        plot_data = plot_data.reset_index()
        plot_data.columns = plot_data.columns.astype(str)

        chart = (
            alt.Chart(plot_data)
            .mark_point()
            .encode(
                x=str(variables[0]), y=str(variables[1]), tooltip=plot_data.columns[0]
            )
            .interactive()
        )

        return chart

except ImportError:
    warnings.warn("Altair is not installed. Interactive plots are not available")
