import numpy as np
from scipy.spatial.distance import cdist

def gaussian(
    X1: np.ndarray,
    X2: np.ndarray,
    sigma: float
) -> float:
    r""" Compute the Gaussian Kernel. 
    Args:
        X1 (np.ndarray): Batch of 1-D array data vectors.
        X2 (np.ndarray): Batch of 1-D array data vector.
        sigma (float): Standard deviation of the gaussian function.
    Returns:
        (np.ndarray): The kernel matrix.
    """

    pairwise_distances = np.array(cdist(X1, X2, 'euclidean'))
    K = np.exp(-pairwise_distances**2 / (2 * sigma**2))
    return np.array(K)