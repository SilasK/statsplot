import pandas as pd
import numpy as np
from scipy import stats


import logging

logger = logging.getLogger("statsplot")


def correct_pvalues_for_multiple_testing(
    p_values, correction_type="Benjamini-Hochberg"
):
    """
    correction_type: one of "Bonferroni", "Bonferroni-Holm", "Benjamini-Hochberg"
    consistent with R - print correct_pvalues_for_multiple_testing([0.0, 0.01, 0.029, 0.03, 0.031, 0.05, 0.069, 0.07, 0.071, 0.09, 0.1])
    """
    from numpy import array, empty, isnan, where

    # remove na values convert to array, store indexes of non NA
    p_values_with_nan = array(p_values, dtype=float)

    not_na_positions = where(~isnan(p_values_with_nan))[0]

    pvalues = p_values_with_nan[not_na_positions]

    # sort p vlaues and prepare unsort index
    sort_index = np.argsort(pvalues)
    pvalues = pvalues[sort_index]
    unsort_index = np.argsort(sort_index)

    n = pvalues.shape[0]
    new_pvalues = empty(n)
    n = float(n)

    if correction_type == "Bonferroni":
        new_pvalues = n * pvalues
    elif correction_type == "Bonferroni-Holm":
        values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
        values.sort()
        for rank, vals in enumerate(values):
            pvalue, i = vals
            new_pvalues[i] = (n - rank) * pvalue
    elif correction_type == "Benjamini-Hochberg":
        values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
        values.sort()
        values.reverse()
        new_values = []
        for i, vals in enumerate(values):
            rank = n - i
            pvalue, index = vals
            new_values.append((n / rank) * pvalue)
        for i in range(0, int(n) - 1):
            if new_values[i] < new_values[i + 1]:
                new_values[i + 1] = new_values[i]
        for i, vals in enumerate(values):
            pvalue, index = vals
            new_pvalues[index] = new_values[i]

    new_pvalues = new_pvalues[unsort_index]

    # add NAn if present

    if len(not_na_positions) < len(p_values_with_nan):
        logger.warn(
            f"{len(p_values_with_nan) - len(not_na_positions)} p values are NA, I don't take them into account"
        )

    corrected_p_values_wiht_na = np.empty_like(p_values_with_nan) * np.nan
    corrected_p_values_wiht_na[not_na_positions] = new_pvalues

    return corrected_p_values_wiht_na


def __stats_test_all_on_once(values1, values2, test, **test_kws):
    ResultsDB = pd.DataFrame(index=values1.columns)
    res = test(values1, values2, **test_kws)
    ResultsDB["Statistic"] = res.statistic
    ResultsDB["Pvalue"] = res.pvalue
    return ResultsDB


def __stats_test_per_value(values1, values2, test, **test_kws):

    Pairwise_comp = pd.DataFrame(columns=["Statistic", "Pvalue"], index=values1.columns)

    for variable in values1.columns:
        try:

            if np.isnan(values1[variable]).any() or np.isnan(values2[variable]).any():

                raise ValueError("Some values are Nan")

            Pairwise_comp.loc[variable] = test(
                values1[variable], values2[variable], **test_kws
            )
        except ValueError:
            Pairwise_comp.loc[variable] = np.nan, np.nan

    return Pairwise_comp


def two_group_test(
    data,
    test_variable,
    comparisons=None,
    ref_group=None,
    test="ttest_ind",
    test_kws=None,
    correct_for_multiple_testing=True,
):

    """test: a parwise statistical test found in scipy e.g ['mannwhitneyu','ttest_ind']
    or a function wich takes two argumens. Additional keyword arguments can be specified by test_kws"""

    # Define test
    if test_kws is None:
        test_kws = {}
    # If test is a string it should be a function found in scipy or 'welch'
    if type(test) == str:

        if test == "welch":
            test_kws.update(dict(equal_var=False))
            test = "ttest_ind"

        assert hasattr(stats, test), "test: {} is not found in scipy".format(test)
        scipy_test = getattr(stats, test)

        if test in ["ttest_ind"]:

            Test = lambda values1, values2: __stats_test_all_on_once(
                values1, values2, scipy_test, **test_kws
            )

        else:
            Test = lambda values1, values2: __stats_test_per_value(
                values1, values2, scipy_test, **test_kws
            )
    # alternatively the test can be a callable
    elif callable(test):
        Test = test
    else:
        Exception(
            "Test should be a string or a callable function got {}".format(type(test))
        )

    data = pd.DataFrame(data)

    Groups = np.unique(test_variable)

    # min value for logFC caluclation
    min_value = data.min().min()

    if min_value == 0:

        log_delta = data.values[data > 0].min() * 0.65
    elif min_value > 0:
        log_delta = 0
    else:

        logger.info("lowest value is negative I don't calculate log2 Fold change")

    if ref_group is not None:
        assert ref_group in Groups, "ref_group: {} is not in the groups: {}".format(
            ref_group, Groups
        )

        Combinations = [(ref_group, other) for other in Groups[Groups != ref_group]]

    elif comparisons is not None:
        Combinations = comparisons

        for c in comparisons:
            assert c[0] in Groups, f"{c[0]} is not in the test variable"
            assert c[1] in Groups, f"{c[1]} is not in the test variable"

    # all combinations
    else:
        from itertools import combinations

        Combinations = list(combinations(Groups, 2))

    Results = {}
    for group1, group2 in Combinations:

        values1 = data.loc[test_variable == group1, :]
        values2 = data.loc[test_variable == group2, :]

        Pairwise_comp = Test(values1, values2)

        Pairwise_comp["median_diff"] = values2.median() - values1.median()

        if min_value >= 0:
            Pairwise_comp["log2FC"] = np.log2(values2.mean() + log_delta) - np.log2(
                values1.mean() + log_delta
            )

        if correct_for_multiple_testing:

            Pairwise_comp["pBH"] = Pairwise_comp[["Pvalue"]].apply(
                correct_pvalues_for_multiple_testing,
                axis=0,
                correction_type="Benjamini-Hochberg",
            )

        Results[group2 + "_vs_" + group1] = Pairwise_comp

    Results = pd.concat(Results, axis=1)
    Results.columns = Results.columns.swaplevel(0, -1)
    Results.sort_index(axis=1, inplace=True)

    return Results


def __apply_to_subsets(data, grouping_variable, function, **kws):

    results = {}

    for subset, subset_data in data.groupby(grouping_variable):
        results[subset] = function(subset_data, **kws)
    return results


def calculate_stats(
    data,
    test_variable,
    grouping_variable=None,
    comparisons=None,
    ref_group=None,
    test="ttest_ind",
    **test_kws,
):

    """Calculate pairewise statistical tests optioonally grouped by a grouping variable"""

    kws = dict(
        test_variable=test_variable,
        ref_group=ref_group,
        comparisons=comparisons,
        test=test,
        test_kws=test_kws,
    )

    if grouping_variable is None:
        results = two_group_test(data, **kws)

    else:
        results = __apply_to_subsets(data, grouping_variable, two_group_test, **kws)

        results = pd.concat(results, axis=1)
        results.columns = results.columns.swaplevel(0, 1)
        results.sort_index(axis=1, inplace=True)

    return results
