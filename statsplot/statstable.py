import logging

logger = logging.getLogger("statstable")

from numpy import dtype, unique
import pandas as pd
import matplotlib.pylab as plt

from .stats import two_group_test
from .siglabels import plot_all_sig_labels
from .plot import statsplot, vulcanoplot

import seaborn as sns


def is_anndata(instance):
    """Function to check if an object is a anndata without importing the package"""

    anndata_attr = ["obs", "var", "to_df", "X"]
    return all([hasattr(instance, a) for a in anndata_attr])


def set_string_indexes(df):

    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)


# TODO: groupby with function such as sum


class MetaTable:
    def __check_consistency(self):

        if self.data.shape[0] != self.obs.shape[0]:
            raise Exception("data and obs are not alligned")
        if self.data.shape[1] != self.var.shape[0]:
            raise Exception("data and var are not alligned")

    def __set_names_and_size(self):
        """Set shape and indexes"""
        self.var_names = self.data.columns
        self.obs_names = self.data.index
        self.shape = self.data.shape

    def __init__(self, data, obs=None, var=None) -> None:

        if type(data) == MetaTable:
            self.data = data.data
            self.obs = data.obs
            self.var = data.var

        elif is_anndata(data):
            self.data = data.to_df()
            self.obs = data.obs
            self.var = data.var

        elif type(data) == pd.DataFrame:
            # parse data
            assert data.shape[0] > 0, "data is empty"
            assert data.index.is_unique, "data has duplicate indices"
            assert data.columns.is_unique, "data has duplicate columns"

            self.data = data
            set_string_indexes(self.data)

            # parse obs

            if obs is None:
                self.obs = pd.DataFrame(index=self.data.index)
            elif type(obs) == pd.DataFrame:

                assert obs.index.is_unique, "obs has duplicate indices"
                self.obs = obs
                set_string_indexes(obs)

                if self.obs.shape[0] != self.data.shape[0]:
                    self.obs = self.obs.loc[self.data.index].copy()

            else:
                raise AttributeError("`obs` should be of type DataFrame or None")

            # parse var

            if var is None:
                self.var = pd.DataFrame(index=self.data.columns)
            elif type(var) == pd.DataFrame:

                assert var.index.is_unique, "var has duplicate indices"
                self.var = var
                set_string_indexes(var)

                if self.var.shape[0] != self.data.shape[0]:
                    self.var = self.var.loc[self.data.columns].copy()

            else:
                raise AttributeError("`var` should be of type DataFrame or None")

            self.__check_consistency()

        else:
            raise AttributeError("`data` needst to be one of [pandas.DataFrame,  ")

        # other attributes commmon to all
        self.__set_names_and_size()

        # link functions from self data to self
        functions_to_link = ["mean", "median", "sum", "std"]
        for f in functions_to_link:
            setattr(self, f, getattr(self.data, f))

    def subset(self, index=None, columns=None):
        assert not (
            (index is None) and (columns is None)
        ), "either indexes or columns needs to be given"

        # fill indexes if None
        if columns is None:
            columns = self.var_names
        elif index is None:
            index = self.obs_names

        return MetaTable(
            data=self.data.loc[index, columns],
            obs=self.obs.loc[index],
            var=self.var.loc[columns],
        )

    def groupby(self, groupby, axis=0):
        if axis == 0:
            G = self.obs.groupby(groupby, axis=0)

            for group in G.indices:
                yield (group, self.subset(index=self.obs_names[G.indices[group]]))

        elif axis == 1:
            # Group by on axis 0. var indexes contain data.columns
            G = self.var.groupby(groupby, axis=0)
            for group in G.indices:
                yield group, self.subset(columns=self.var_names[G.indices[group]])
        else:
            raise Exception("axis should be 1 or 2")

    def __repr__(self):
        value = f"MetaTable with {self.shape[0]} samples x {self.shape[1]} features\n"
        f"Sample annotations: {list(self.obs.columns)}\n"
        f"Feature annotations: {list(self.var.columns)} "
        return value


class StatsTable(MetaTable):
    def __init__(
        self,
        data,
        test_variable,
        comparisons=None,
        ref_group=None,
        test="welch",
        test_kws=None,
        grouping_variable=None,
        order_grouping=None,
        order_test=None,
        colors=None,
        data_unit=None,
        label_variable=None,
    ):

        # create MetaTable object
        super().__init__(data)

        if type(test_variable) == str:
            self.test_variable = self.obs[test_variable]

        elif type(test_variable) == pd.Series:

            self.obs = self.obs.join(test_variable)
            self.test_variable = self.obs[test_variable.name]

        else:

            raise IOError(
                "Input type data and test_variable not consistent. Either data is an AnData     obejct and test_ariable a string refering to a coulmn in data.obs. or test_variable is a pandas series."
            )

        if order_test is None:
            order_test = unique(self.test_variable)
        self.order_test = order_test

        if colors is None:
            self.colors = sns.color_palette("deep", len(order_test))
        else:
            assert len(colors) == len(self.order_test)
            self.colors = colors

        if grouping_variable is not None:

            assert (
                grouping_variable in self.obs.columns
            ), f"{grouping_variable} is not in the metadata columns: {self.obs.columns}"

            self.grouping_variable = self.obs[grouping_variable]

            if order_grouping is None:
                order_grouping = unique(self.grouping_variable)

            self.order_grouping = order_grouping
        else:
            self.order_grouping, self.grouping_variable = None, None

        self.data_unit = data_unit

        if label_variable is None:
            self.labels = pd.Series(self.var_names, self.var_names)

        elif type(label_variable) == str:

            self.labels = self.var[label_variable]

        elif type(label_variable) == pd.Series:
            if label_variable.name is None:
                label_variable.name = "Label"
            else:
                assert (
                    label_variable.name not in self.var.columns
                ), "Label is already found in var"

            self.var[label_variable.name] = label_variable
            self.labels = self.var[label_variable.name]
        else:
            raise IOError("label_variable must be None, a string, or a pandas series")

        if not self.labels.is_unique:
            logger.warn(
                "Your labels are not unique. but I should be able to handle this."
            )

        self.__calculate_stats__(
            test=test, test_kws=test_kws, comparisons=comparisons, ref_group=ref_group
        )

    def __repr__(self) -> str:
        annadata_str = super().__repr__()
        annadata_str += f"\n    test_variable: {self.test_variable.name} with groups {self.order_test}  "

        if self.grouping_variable is None:
            annadata_str += "\n    grouping_variable: None"
        else:
            annadata_str += f"\n    grouping_variable: {self.grouping_variable.name} with groups {self.order_grouping}  "

        return annadata_str

    def __apply_to_subsets(self, function, **kws):

        assert self.grouping_variable is not None

        results = {}

        for subset, subset_metatable in self.groupby(self.grouping_variable):
            results[subset] = function(subset_metatable.data, **kws)
        return results

    def __calculate_stats__(
        self, comparisons=None, ref_group=None, test="welch", **test_kws
    ):

        if ref_group is not None:
            assert ref_group in self.order_test, f"{ref_group} is not in the order_test"

        elif comparisons is not None:
            for c in comparisons:
                assert (
                    c[0] in self.order_test
                ), f"{c[0]} is not in the test variable or order_test"
                assert (
                    c[1] in self.order_test
                ), f"{c[1]} is not in the test variabl or order_test"

        function = two_group_test

        kws = dict(
            test_variable=self.test_variable,
            ref_group=ref_group,
            comparisons=comparisons,
            test=test,
            **test_kws,
        )

        if self.grouping_variable is None:
            results = function(self.data, **kws)

        else:
            results = self.__apply_to_subsets(function, **kws)

            results = pd.concat(results, axis=1)
            results.columns = results.columns.swaplevel(0, 1)
            results.sort_index(axis=1, inplace=True)

        ## Add description to stats

        description = self.var.copy()

        if len(results.columns.levshape) == 3:

            description.columns = pd.MultiIndex.from_arrays(
                [
                    ["Description"] * description.shape[1],
                    ["All"] * description.shape[1],
                    description.columns,
                ]
            )
        elif len(results.columns.levshape) == 2:
            description.columns = pd.MultiIndex.from_arrays(
                [["Description"] * description.shape[1], description.columns]
            )

        self.stats = results.astype(float).join(description)

    def plot(
        self,
        variable,
        show_dots=True,
        distance_between_sig_labels="auto",
        box_params=None,
        swarm_params=None,
        corrected_pvalues=False,
        show_not_significant=False,
        ax=None,
        **labelkws,
    ):

        labelkws["deltay"] = distance_between_sig_labels

        if ax is None:
            ax = plt.subplot(111)

        if corrected_pvalues:
            p_value_name = "pBH"
        else:
            p_value_name = "Pvalue"

        statsplot(
            self.data[variable],
            self.test_variable,
            order_test=self.order_test,
            grouping_variable=self.grouping_variable,
            order_grouping=self.order_grouping,
            show_dots=show_dots,
            box_params=box_params,
            swarm_params=swarm_params,
            show_not_significant=show_not_significant,
            labelkws=labelkws,
            palette=self.colors,
            p_values=self.stats[p_value_name].loc[variable].T,
            ax=ax,
        )

        ax.set_title(self.labels[variable])
        ax.set_ylabel(self.data_unit)

    def __get_groups(self, subset=None):
        "Check if given subset are in header of statstable"
        "Otherwise return all in a row, if not defined return None"

        # check if grouped
        if self.grouping_variable is None:
            return None
        else:

            all_groups = list(self.stats.Pvalue.columns.get_level_values(-2).unique())
            if subset is None:

                return all_groups
            elif type(subset) == str:
                assert subset in all_groups, f"{g} is not in the Groups"

                return [subset]
            else:
                for g in subset:
                    assert g in all_groups, f"{g} is not in the Groups"

                return list(subset)

    def __get_comparisons(self, subset=None):
        "Check if given subset are in comparisons of statstable"
        "Otherwise return all in a row, if not defined return None"

        all_comparisons = list(self.stats.Pvalue.columns.get_level_values(-1).unique())
        if subset is None:

            return all_comparisons
        elif type(subset) == str:
            assert subset in all_comparisons, f"{g} is not in the Comparisons"
            return [subset]
        else:
            for g in subset:
                assert g in all_comparisons, f"{g} is not in the Comparisons"

            return list(subset)


# TODO: Hide output axes labesls
    def vulcanoplot(
        self,
        comparisons=None,
        groups=None,
        corrected_pvalues=False,
        threshold_p=None,
        hue=None,
        figsize=(6, 6),
        label_points="auto",
        max_labels=5,
        effect_label=None,
        pvalue_label=None,
        **kws,
    ):

        if "log2FC" in self.stats.columns:
            effect_name = "log2FC"
            x_label = "$\log_2FC$"

        else:
            effect_name = "median_diff"
            x_label = "median difference"
            logger.info("Don't have log2FC in stats, using median_diff for vulcanoplot")

        if effect_label is not None:
            x_label = effect_label

        def rename_vulcano_axis_labels():
            ax = plt.gca()
            ax.set_xlabel(x_label)

            if pvalue_label is not None:
                ax.set_ylabel(pvalue_label)
            elif corrected_pvalues:
                ax.set_ylabel("$-\log(P_{BH})$")
            # ellse default label from vulcano plot

        groups = self.__get_groups(groups)

        comparisons = self.__get_comparisons(comparisons)

        axes = []

        if corrected_pvalues:
            p_value_name = "pBH"
            if threshold_p is None:
                threshold_p = 0.1
        else:
            p_value_name = "Pvalue"
            if threshold_p is None:
                threshold_p = 0.05

        if hue is not None and (type(hue) == str):
            hue = self.var[hue]

        # collect general arguments
        kws["threshold_p"] = threshold_p
        kws["hue"] = hue
        kws["labels"] = self.labels

        # map kws to vulcanoplot
        kws["label_points"] = label_points
        kws["max_labels"] = max_labels
        kws["figsize"] = figsize

        if groups is not None:

            for g in groups:
                for c in comparisons:

                    vulcanoplot(
                        p_values=self.stats[p_value_name][g][c],
                        effect=self.stats[effect_name][g][c],
                        **kws,
                    )
                    ax = plt.gca()
                    ax.set_title(g)
                    rename_vulcano_axis_labels()
                    axes.append(ax)

        else:

            for c in comparisons:

                vulcanoplot(
                    p_values=self.stats[p_value_name][c],
                    effect=self.stats[effect_name][c],
                    **kws,
                )
                rename_vulcano_axis_labels()
                axes.append(plt.gca())

        return axes
