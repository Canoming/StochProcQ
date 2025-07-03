"""
This module contains functions to compute various measures between probability distributions.
"""

import numpy as np

def kl_div(p:np.ndarray, q:np.ndarray) -> float:
    """
    Compute the Kullback-Leibler divergence between two probability distributions.

    Parameters:
    -------------------
    p: np.ndarray
        The first probability distribution.
    q: np.ndarray
        The second probability distribution.

    Return:
    ----------------
    kl: float
        The Kullback-Leibler divergence between the two distributions.
    """
    return np.sum(p * np.log(p / q), axis=-1)
