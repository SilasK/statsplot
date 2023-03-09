


from numpy import log
import pandas as pd
import numpy as np


# copied from scikit-bio 
# because I cannot install it
def closure(mat):
    """
    Performs closure to ensure that all elements add up to 1.
    Parameters
    ----------
    mat : array_like
       a matrix of proportions where
       rows = compositions
       columns = components
    Returns
    -------
    array_like, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1
    Raises
    ------
    ValueError
       Raises an error if any values are negative.
    ValueError
       Raises an error if the matrix has more than 2 dimension.
    ValueError
       Raises an error if there is a row that has all zeros.
    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import closure
    >>> X = np.array([[2, 2, 6], [4, 4, 2]])
    >>> closure(X)
    array([[ 0.2,  0.2,  0.6],
           [ 0.4,  0.4,  0.2]])
    """
    mat = np.atleast_2d(mat)
    if np.any(mat < 0):
        raise ValueError("Cannot have negative proportions")
    if mat.ndim > 2:
        raise ValueError("Input matrix can only have two dimensions or less")
    if np.all(mat == 0, axis=1).sum() > 0:
        raise ValueError("Input matrix cannot have rows with all zeros")
    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat.squeeze()


def multiplicative_replacement(mat, delta=None):
    r"""Replace all zeros with small non-zero values
    It uses the multiplicative replacement strategy [1]_ ,
    replacing zeros with a small positive :math:`\delta`
    and ensuring that the compositions still add up to 1.
    Parameters
    ----------
    mat: array_like
       a matrix of proportions where
       rows = compositions and
       columns = components
    delta: float, optional
       a small number to be used to replace zeros
       If delta is not specified, then the default delta is
       :math:`\delta = \frac{1}{N^2}` where :math:`N`
       is the number of components
    Returns
    -------
    numpy.ndarray, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1
    Raises
    ------
    ValueError
       Raises an error if negative proportions are created due to a large
       `delta`.
    Notes
    -----
    This method will result in negative proportions if a large delta is chosen.
    References
    ----------
    .. [1] J. A. Martin-Fernandez. "Dealing With Zeros and Missing Values in
           Compositional Data Sets Using Nonparametric Imputation"
    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import multiplicative_replacement
    >>> X = np.array([[.2,.4,.4, 0],[0,.5,.5,0]])
    >>> multiplicative_replacement(X)
    array([[ 0.1875,  0.375 ,  0.375 ,  0.0625],
           [ 0.0625,  0.4375,  0.4375,  0.0625]])
    """
    mat = closure(mat)
    z_mat = (mat == 0)

    num_feats = mat.shape[-1]
    tot = z_mat.sum(axis=-1, keepdims=True)

    if delta is None:
        delta = (1. / num_feats)**2

    zcnts = 1 - tot * delta
    if np.any(zcnts) < 0:
        raise ValueError('The multiplicative replacement created negative '
                         'proportions. Consider using a smaller `delta`.')
    mat = np.where(z_mat, delta, zcnts * mat)
    return mat.squeeze()



def clr(data: pd.DataFrame, log=log,features="all"):
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
    matrix = multiplicative_replacement(matrix)

    # CLR
    matrix = log(matrix)

    # Center
    if features.lower() == "all":

        mean = matrix.mean(1)

    elif features.lower() == "nz":
            
        mean = matrix[matrix != 0].mean(1)
    elif features.lower() == "iql":
        # use mean of features in interquartile range
        q1 = matrix.quantile(0.25, axis=1)
        q3 = matrix.quantile(0.75, axis=1)
        mean = matrix[(matrix > q1) & (matrix < q3)].mean(1)
    else:
        raise Exception("features must be 'all', 'nz', or 'iql'")


    matrix = (matrix.T - mean).T

    if type(data) == pd.DataFrame:

        return pd.DataFrame(matrix, index=d.index, columns=d.columns)
