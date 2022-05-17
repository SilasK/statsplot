import imp
import logging

from matplotlib.pyplot import violinplot

logger = logging.getLogger("statstable")

from anndata import AnnData
from numpy import unique
import pandas as pd
import matplotlib.pylab as plt

from .stats import two_group_test
from .siglabels import plot_all_sig_labels
from .plot import statsplot, vulcanoplot

import seaborn as sns



class StatsTable(AnnData):
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

        # create AnnData object
        super().__init__(data, obs=None, var=None)

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

        self.__calculate_stats__(
            test=test, test_kws=test_kws, comparisons=comparisons, ref_group=ref_group
        )

        self.data_unit = data_unit

        if label_variable is None:
            self.labels = pd.Series(self.var_names, self.var_names)

        elif type(label_variable) == str:
            self.labels = self.var[label_variable]
        elif type(label_variable) == pd.Series:

            self.var = self.var.join(label_variable)
            self.labels = self.var[label_variable.name]
        else:
            raise IOError("label_variable must be None, a string, or a pandas series")

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

        for subset, subset_data in self.to_df().groupby(self.grouping_variable):
            results[subset] = function(subset_data, **kws)
        return results

    def __calculate_stats__(
        self, comparisons=None, ref_group=None, test="welch", test_kws=None
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

        if test_kws is None:
            test_kws = {}

        kws = dict(
            test_variable=self.test_variable,
            ref_group=ref_group,
            comparisons=comparisons,
            test=test,
            **test_kws,
        )

        if self.grouping_variable is None:
            results = function(self.to_df(), **kws)

        else:
            results = self.__apply_to_subsets(function, **kws)

            results = pd.concat(results, axis=1)
            results.columns = results.columns.swaplevel(0, 1)
            results.sort_index(axis=1, inplace=True)

        self.stats = results.astype(float)



    def plot(
        self,
        variable,
        distance_between_sig_labels="auto",
        box_params=None,
        swarm_params=None,
        ax=None,
        **labelkws,
    ):

        labelkws["deltay"] = distance_between_sig_labels

        if ax is None:
            ax = plt.subplot(111)

        statsplot(
            self[:, variable].to_df()[variable],
            self.test_variable,
            order_test=self.order_test,
            grouping_variable=self.grouping_variable,
            order_grouping=self.order_grouping,
            box_params=box_params,
            swarm_params=swarm_params,
            labelkws=labelkws,
            palette=self.colors,
            p_values=self.stats.Pvalue.loc[variable].T,
            ax=ax,
        )

        ax.set_title(self.labels[variable])
        ax.set_ylabel(self.data_unit)

    def __get_groups(self, subset = None):
        "Check if given subset are in header of statstable"
        "Otherwise return all in a row, if not defined return None"

            # check if grouped
        if self.grouping_variable is None:
            return None
        else:

            all_groups = list(self.stats.columns.levels[-2])
            if subset is None:

                return all_groups
            elif type(subset) == str:
                return [subset]
            else:
                for g in subset:
                    assert g in all_groups, f"{g} is not in the Groups"
                
                return list(subset)

    def __get_comparisons(self, subset = None):
        "Check if given subset are in comparisons of statstable"
        "Otherwise return all in a row, if not defined return None"


        all_comparisons = list(self.stats.columns.levels[-1])
        if subset is None:

            return all_comparisons
        elif type(subset) == str:
            return [subset]
        else:
            for g in subset:
                assert g in all_comparisons, f"{g} is not in the Comparisons"
            
            return list(subset)

    
  


    def vulcanoplot(self, comparisons = None, groups = None, corrected_pvalues= True, threshold_p= None , **kws):

        if "log2FC" in self.stats.columns:
            effect_name= "log2FC"

        else:
            effect_name= "median_diff"
            logger.info("Don't have log2FC in stats, using median_diff for vulcanoplot")

        groups = self.__get_groups(groups)
     

        comparisons = self.__get_comparisons(comparisons)


        if corrected_pvalues:
            p_value_name= 'pBH'
            if threshold_p is None:
                threshold_p= 0.1
        else:
            p_value_name = "Pvalue"
            if threshold_p is None:
                threshold_p= 0.05
      
        if groups is not None:

            for g in groups:
                for c in comparisons:

                    
                    vulcanoplot( p_values=self.stats[p_value_name][g][c], 
                                      effect= self.stats[effect_name][g][c], 
                                      threshold_p= threshold_p,
                                      **kws
                                     )
                    ax= plt.gca()
                    ax.set_title(g)
        else:

          
            for c in comparisons:

                vulcanoplot( p_values=self.stats[p_value_name][c], 
                                    effect= self.stats[effect_name][c], 
                                    threshold_p= threshold_p,
                                    **kws
                                    )


