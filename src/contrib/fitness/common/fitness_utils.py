
from typing import List
import numpy as np


def get_counts_of_array(arr: np.ndarray, ignore_zero=True) -> List[float]:
    """Returns a list of counts of the unique elements in an array

    Args:
        arr (np.ndarray): The array to count
        ignore_zero (bool, optional): If this is true, zeros are not counted. Defaults to True.

    Returns:
        List[float]: 
    """
    uniques, counts = np.unique(arr, return_counts=True)
    if uniques[0] == 0 and ignore_zero: counts = counts[1:] 
    return counts


if __name__ == '__main__':
    # test quick
    print(get_counts_of_array([0, 1, 2, 1, 2, 0]))
    print(get_counts_of_array([0, 1, 2, 1, 2, 0], ignore_zero=False))

    print(get_counts_of_array([1, 2, 1, 2,]))
    print(get_counts_of_array([1, 2, 1, 2,], ignore_zero=False))