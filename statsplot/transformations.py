try:
    from skbio.stats import composition
except ImportError as e:
    raise Exception(
        "'scikit-bio' is required for this sub-package. Install id with pip or conda"
    ) from e


from numpy import log
import pandas as pd


from typing import Union


def clr(data: pd.DataFrame, log=log):
    """
    Centered log ratio (CLR) with multiplicative replacement implemented in scikit-bio
    """

    if type(data) == pd.DataFrame:

        # remove columns with all zeros

        d = data.loc[:, ~(data == 0).all()]
        # get data as matrix
        matrix = d.values
    else:
        raise Exception("data must be a pandas.DataFrame")

    # Fill in zeros with multiplicative replacement
    matrix = composition.multiplicative_replacement(matrix)

    # CLR
    matrix = log(matrix)

    matrix = (matrix.T - matrix.mean(1)).T

    if type(data) == pd.DataFrame:

        return pd.DataFrame(matrix, index=d.index, columns=d.columns)
