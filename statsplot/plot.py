import matplotlib.pyplot as plt
import seaborn as sns
from numpy import unique, log10,abs
import pandas as pd

from .stats import  calculate_stats
from .siglabels import plot_all_sig_labels

def vulcanoplot(p_values,effect,hue = None,threshold_p = 0.05, figsize=(4,4),**kws):

    f= plt.figure(figsize=figsize)


    assert effect.ndim==1, "effect should be one dimensional"
    assert p_values.ndim==1, "p_values should be one dimensional"

    
    logPvalues = - log10( p_values.astype(float) )
    logPvalues.name= "$-\log(P)$"

    threshold = - log10(threshold_p)

    if hue is None:
        hue= pd.Series("Significant", index= logPvalues.index)


    ax=sns.scatterplot(y= logPvalues.loc[logPvalues> threshold] ,x= effect,
                   hue= hue.loc[logPvalues> threshold])

    ax=sns.scatterplot(y= logPvalues.loc[logPvalues<= threshold] ,x= effect,
                       color='grey',marker='.',
                       label='not significant'
                   )




    ax.legend(bbox_to_anchor=(1,1)) 
    ax_lim= abs(ax.get_xlim()).max()
    ax.set_xlim([-ax_lim,ax_lim])

    if '_vs_' in effect.name:
        g1,g2 = effect.name.split('_vs_')

        ax.annotate(g1, (ax_lim*0.9,0), ha='right')
        ax.annotate(g2, (-ax_lim*0.9,0), ha='left')
        
        

def statsplot(
    variable,
    test_variable,
    data = None,
    order_test=None,
    grouping_variable=None,
    order_grouping=None,
    box_params=None,
    swarm_params=None,
    labelkws=None,
    stats_kws=None,
    palette=None,
    p_values=None,
    test="ttest_ind",
    ax=None,
):

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

    legend = ax.get_legend_handles_labels()

    sns.swarmplot(**params, color="k", dodge=True, **swarm_params)

    if grouping_variable is not None:
        ax.legend(*legend, bbox_to_anchor=(1, 1))

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

    plot_all_sig_labels(p_values, order_test, order_grouping, ax=ax, **labelkws)

    return ax, p_values

