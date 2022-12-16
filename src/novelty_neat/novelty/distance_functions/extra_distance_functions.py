import numpy as np
from scipy.spatial import distance

def manhattan(a: np.ndarray, b: np.ndarray) -> float:
    return np.abs(a-b).sum()

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten()
    b = b.flatten()
    return np.argmax(a.dot(b)/ (np.linalg.norm(a, axis=0) * np.linalg.norm(b)))
  

def chebyshev(a: np.ndarray, b: np.ndarray) -> float:
    return np.max(np.abs(a-b))

def minkowski(a: np.ndarray, b: np.ndarray) -> float:
        a = a.flatten()
        b = b.flatten()
        return distance.minkowski(a, b, p=2, w=None)

# other functions to test -- coming soon

# def Jaccard(a: np.ndarray, b: np.ndarray) -> float:
#     return np.abs(a-b).sum()
#     # return np.linalg.norm(a-b, ord=1)


# def Haversine(a: np.ndarray, b: np.ndarray) -> float:
#     return np.abs(a-b).sum()
#     # return np.linalg.norm(a-b, ord=1)

 
# . Sørensen-Dice

