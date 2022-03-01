

try:
    from skbio.stats import composition
except ImportError as e:
    raise Exception("'scikit-bio' is required for this sub-package. Install id with pip or conda") from e


from numpy import log
import pandas as pd
import anndata

from typing import Union

def clr(data : Union[pd.DataFrame, anndata.AnnData] ,log=log):
    """
        Centered log ratio (CLR) with multiplicative replacement implemented in scikit-bio
    """

    if type(data) == pd.DataFrame:

        # remove columns with all zeros

        d = data.loc[:,~(data==0).all()]
        # get data as matrix
        matrix = d.values
    elif type(data) == anndata.AnnData:
        # remove columns with all zeros
        d = data[:,~(data.X==0).all(axis=0)].copy()
        # get data as matrix
        matrix = d.X
    else:
        raise Exception("data must be a pandas.DataFrame or anndata.AnnData")


    # Fill in zeros with multiplicative replacement
    matrix = composition.multiplicative_replacement(matrix)


    # CLR
    matrix = log(matrix)

    matrix = (matrix.T - matrix.mean(1)).T

    if type(data) == pd.DataFrame:

        return pd.DataFrame(matrix, index=d.index, columns=d.columns)

    elif type(data) == anndata.AnnData:
        d.X= matrix
        return d

  
