import numpy as np
from sklearn.utils import check_random_state


def generate_sample_indices(random_state, n_samples):
    """
    Generate indices for random samples from a dataset.

    This function creates a random sample of indices based on the specified
    number of samples in the dataset. It uses a provided random state for
    reproducibility.

    Parameters
    ----------
    random_state : int, RandomState instance or None
        The generator used to initialize the random state. If int, random_state is the seed used
        by the random number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the RandomState instance used
        by np.random.

    n_samples : int
        The number of samples to generate indices for.

    Returns
    -------
    sample_indices : ndarray of shape (n_samples,)
        The generated random sample indices.

    Examples
    --------
    >>> generate_sample_indices(random_state=42, n_samples=5)
    array([3, 4, 2, 2, 2])
    """
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)
    return sample_indices


def generate_unsampled_indices(random_state, n_samples):
    """
    Generate indices for samples that are not selected (out-of-bag samples).

    This function identifies the indices of samples that are not included in
    a random sample of the dataset, effectively identifying out-of-bag samples.

    Parameters
    ----------
    random_state : int, RandomState instance or None
        The generator used to initialize the random state. If int, random_state is the seed used
        by the random number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the RandomState instance used
        by np.random.

    n_samples : int
        The total number of samples in the dataset.

    Returns
    -------
    unsampled_indices : ndarray
        Indices of the samples that were not selected in the random sample.

    Examples
    --------
    >>> generate_unsampled_indices(random_state=42, n_samples=5)
    array([0, 1, 3])
    """
    sample_indices = generate_sample_indices(random_state, n_samples)
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]
    return unsampled_indices
