import numpy as np

DATA = [
np.array([[0, 1, 1, 1, 0],
       [0, 0, 1, 1, 1],
       [0, 1, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [1, 0, 0, 0, 0]]),
np.array([[1, 0, 1, 1, 0],
       [1, 0, 0, 1, 1],
       [1, 0, 0, 1, 0],
       [0, 0, 0, 1, 0],
       [1, 0, 1, 1, 0]]),
np.array([[1, 1, 1, 1, 1],
       [1, 0, 0, 0, 0],
       [0, 0, 1, 0, 0],
       [1, 0, 1, 0, 1],
       [1, 0, 0, 1, 1]]),
np.array([
       [1, 1, 0, 0, 0],
       [1, 1, 0, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 1],
       [0, 1, 0, 0, 0]]),
np.array([[0, 1, 0, 1, 1],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 1, 1],
       [1, 1, 0, 1, 1],
       [1, 1, 0, 0, 0]]),
np.array([[0, 1, 1, 0, 1],
       [0, 0, 0, 0, 1],
       [0, 1, 1, 0, 1],
       [1, 0, 0, 0, 0],
       [0, 1, 1, 0, 0]]),
np.array([[1, 0, 0, 0, 1],
       [0, 1, 1, 0, 1],
       [0, 1, 1, 1, 0],
       [1, 0, 0, 0, 1],
       [1, 0, 1, 0, 1]]),
np.array([[0, 1, 0, 1, 0],
       [0, 0, 1, 1, 0],
       [1, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 1]]),
np.array([[1, 0, 0, 1, 1],
       [0, 0, 1, 0, 0],
       [1, 1, 1, 0, 1],
       [0, 0, 1, 1, 1],
       [1, 1, 0, 0, 0]]),
np.array([[0, 1, 0, 1, 1],
       [1, 1, 1, 0, 0],
       [1, 0, 0, 1, 0],
       [1, 1, 1, 0, 0],
       [1, 0, 1, 1, 1]]),
np.array([[0, 0, 1, 0, 0],
       [1, 1, 1, 1, 0],
       [1, 1, 0, 1, 1],
       [1, 1, 1, 0, 0],
       [1, 1, 1, 1, 1]]),
np.array([[1, 1, 1, 1, 1],
       [1, 0, 0, 0, 1],
       [0, 0, 1, 0, 1],
       [0, 0, 1, 0, 0],
       [1, 1, 0, 1, 0]]),
np.array([[0, 1, 1, 1, 1],
       [1, 1, 0, 1, 0],
       [0, 0, 0, 1, 1],
       [1, 0, 0, 0, 0],
       [1, 1, 1, 0, 1]]),
np.array([[1, 1, 1, 0, 0],
       [0, 1, 0, 0, 0],
       [0, 1, 1, 0, 0],
       [0, 0, 0, 0, 0],
       [1, 1, 0, 0, 1]]),
np.array([[1, 1, 1, 0, 0],
       [1, 0, 1, 1, 1],
       [0, 1, 0, 1, 1],
       [0, 1, 1, 0, 0],
       [0, 0, 0, 1, 0]]),
np.array([[0, 1, 0, 1, 0],
       [0, 1, 1, 1, 1],
       [1, 0, 0, 1, 1],
       [1, 0, 0, 1, 0],
       [1, 1, 0, 1, 1]]),
np.array([[0, 1, 1, 1, 0],
       [1, 0, 0, 1, 0],
       [0, 1, 1, 1, 1],
       [0, 0, 0, 1, 1],
       [1, 1, 0, 0, 0]]),
np.array([[1, 0, 1, 0, 0],
       [0, 1, 1, 0, 0],
       [1, 0, 1, 1, 0],
       [0, 1, 0, 0, 1],
       [1, 1, 1, 0, 0]]),
np.array([[1, 1, 0, 1, 0],
       [0, 0, 0, 0, 0],
       [0, 1, 1, 1, 1],
       [0, 0, 1, 0, 1],
       [0, 1, 0, 1, 1]]),
np.array([[1, 0, 1, 1, 1],
       [1, 1, 0, 1, 1],
       [1, 0, 0, 1, 1],
       [0, 1, 0, 0, 0],
       [0, 1, 0, 1, 1]]),
np.array([[1, 1, 0, 1, 1],
       [0, 1, 1, 1, 1],
       [1, 0, 1, 1, 0],
       [0, 0, 1, 1, 0],
       [1, 0, 1, 1, 1]]),
np.array([[0, 0, 0, 0, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 0, 0, 0],
       [1, 1, 0, 1, 1],
       [0, 0, 1, 1, 1]]),
np.array([[1, 1, 1, 1, 1],
       [0, 0, 0, 1, 1],
       [1, 0, 0, 0, 0],
       [1, 0, 0, 1, 0],
       [1, 1, 0, 1, 1]]),
np.array([[1, 0, 1, 0, 0],
       [0, 1, 0, 0, 1],
       [0, 0, 0, 0, 0],
       [0, 1, 0, 0, 1],
       [0, 1, 0, 0, 0]]),
np.array([[0, 0, 0, 0, 0],
       [0, 0, 0, 1, 1],
       [0, 1, 1, 0, 1],
       [1, 0, 0, 1, 1],
       [0, 0, 0, 0, 1]]),
np.array([[0, 0, 1, 0, 0],
       [1, 1, 1, 1, 0],
       [0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [1, 1, 1, 1, 0]]),
np.array([[0, 1, 1, 1, 0],
       [0, 0, 0, 1, 1],
       [0, 1, 1, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 1, 0, 1, 0]]),
np.array([[1, 0, 0, 1, 1],
       [1, 0, 0, 1, 1],
       [1, 1, 0, 1, 0],
       [0, 0, 1, 0, 0],
       [1, 0, 1, 0, 0]]),
np.array([[1, 0, 0, 0, 0],
       [1, 0, 1, 0, 1],
       [1, 1, 0, 0, 0],
       [1, 1, 1, 0, 1],
       [1, 0, 0, 1, 1]]),
np.array([[1, 0, 1, 0, 0],
       [0, 1, 1, 1, 0],
       [1, 0, 1, 0, 0],
       [0, 0, 1, 1, 0],
       [1, 0, 1, 1, 1]]),
np.array([[0, 1, 1, 0, 0],
       [1, 1, 1, 1, 1],
       [0, 0, 0, 1, 0],
       [1, 1, 1, 0, 1],
       [0, 1, 0, 0, 1]]),
np.array([[0, 0, 1, 1, 1],
       [1, 1, 0, 0, 0],
       [1, 1, 0, 0, 1],
       [0, 1, 1, 1, 0],
       [1, 1, 1, 1, 1]]),
np.array([[1, 0, 1, 1, 0],
       [1, 1, 1, 0, 0],
       [1, 1, 0, 1, 1],
       [1, 1, 0, 0, 0],
       [1, 0, 0, 0, 1]]),
np.array([[1, 0, 0, 1, 0],
       [0, 0, 0, 0, 0],
       [0, 1, 0, 1, 1],
       [0, 1, 0, 1, 0],
       [0, 0, 1, 1, 1]]),
np.array([[0, 1, 0, 1, 0],
       [0, 0, 0, 1, 0],
       [1, 0, 1, 0, 0],
       [1, 0, 0, 1, 1],
       [0, 0, 1, 0, 0]]),
np.array([[0, 1, 1, 0, 0],
       [0, 1, 0, 0, 1],
       [0, 1, 0, 0, 0],
       [1, 1, 1, 0, 1],
       [0, 0, 1, 0, 0]]),
np.array([[0, 0, 0, 1, 0],
       [1, 1, 0, 0, 0],
       [1, 0, 1, 0, 1],
       [0, 0, 1, 1, 0],
       [1, 0, 1, 1, 0]]),
np.array([[0, 0, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 1, 0, 1],
       [0, 1, 1, 1, 0],
       [0, 0, 1, 1, 1]]),
np.array([[1, 1, 0, 1, 0],
       [0, 1, 0, 0, 1],
       [0, 1, 0, 0, 0],
       [0, 1, 0, 1, 0],
       [1, 1, 1, 0, 0]]),
np.array([[0, 1, 0, 0, 1],
       [0, 1, 0, 0, 1],
       [0, 1, 0, 0, 1],
       [1, 1, 0, 1, 0],
       [0, 1, 0, 1, 0]]),
np.array([[0, 1, 1, 1, 1],
       [1, 1, 1, 0, 0],
       [0, 1, 1, 0, 1],
       [1, 0, 1, 1, 0],
       [1, 0, 1, 1, 1]]),
np.array([[0, 1, 1, 0, 1],
       [0, 1, 0, 0, 0],
       [1, 0, 1, 0, 0],
       [1, 1, 0, 1, 1],
       [1, 0, 0, 1, 0]]),
np.array([[1, 1, 1, 1, 1],
       [0, 1, 0, 0, 1],
       [1, 1, 1, 0, 0],
       [1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1]]),
np.array([[0, 0, 1, 0, 1],
       [1, 0, 0, 1, 1],
       [0, 1, 0, 1, 0],
       [1, 1, 1, 1, 0],
       [0, 0, 0, 1, 0]]),
np.array([[0, 1, 1, 1, 1],
       [0, 0, 1, 1, 0],
       [1, 0, 0, 1, 1],
       [1, 0, 1, 0, 0],
       [1, 1, 1, 1, 1]]),
np.array([[1, 0, 0, 0, 0],
       [1, 0, 0, 0, 1],
       [0, 1, 0, 0, 1],
       [1, 0, 1, 1, 0],
       [1, 1, 1, 1, 1]]),
np.array([[1, 1, 1, 1, 1],
       [1, 0, 1, 1, 0],
       [1, 0, 0, 1, 0],
       [1, 1, 1, 0, 0],
       [1, 1, 1, 0, 1]]),
np.array([[1, 1, 0, 1, 1],
       [1, 0, 1, 1, 0],
       [1, 1, 0, 0, 1],
       [0, 1, 0, 1, 0],
       [1, 1, 1, 1, 0]]),
np.array([[1, 0, 0, 0, 0],
       [1, 1, 0, 0, 1],
       [0, 0, 0, 1, 1],
       [0, 0, 1, 1, 0],
       [1, 0, 0, 1, 0]]),
np.array([[1, 0, 0, 1, 0],
       [1, 0, 0, 0, 1],
       [0, 1, 0, 1, 0],
       [1, 1, 1, 1, 1],
       [1, 0, 0, 1, 1]]),
np.array([[1, 0, 0, 0, 1],
       [0, 0, 0, 0, 1],
       [0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0],
       [1, 1, 1, 0, 0]]),
np.array([[0, 0, 1, 0, 0],
       [1, 1, 1, 0, 0],
       [1, 0, 0, 0, 0],
       [0, 0, 1, 1, 1],
       [0, 0, 0, 1, 1]]),
np.array([[0, 1, 0, 1, 1],
       [1, 1, 1, 0, 1],
       [0, 1, 1, 1, 0],
       [1, 1, 1, 0, 0],
       [0, 0, 1, 0, 1]]),
np.array([[0, 0, 1, 1, 0],
       [1, 1, 0, 1, 1],
       [1, 0, 1, 0, 0],
       [0, 0, 1, 1, 1],
       [0, 0, 0, 1, 0]]),
np.array([[1, 1, 0, 1, 0],
       [1, 1, 0, 0, 1],
       [1, 0, 1, 1, 1],
       [0, 1, 1, 0, 0],
       [1, 0, 1, 0, 0]]),
np.array([[1, 0, 1, 0, 1],
       [0, 0, 1, 0, 1],
       [0, 1, 0, 1, 0],
       [0, 0, 0, 1, 0],
       [1, 0, 0, 1, 1]]),
np.array([[0, 0, 0, 1, 1],
       [0, 1, 1, 1, 1],
       [0, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [0, 1, 0, 0, 1]]),
np.array([[1, 0, 1, 1, 1],
       [0, 1, 0, 1, 0],
       [0, 0, 1, 0, 1],
       [1, 0, 1, 0, 1],
       [0, 1, 1, 1, 1]]),
np.array([[0, 1, 1, 1, 0],
       [1, 1, 1, 0, 1],
       [0, 1, 0, 1, 1],
       [0, 1, 0, 1, 1],
       [1, 1, 1, 0, 0]]),
np.array([[1, 1, 1, 1, 0],
       [1, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 0, 0, 1],
       [1, 1, 0, 0, 1]]),
np.array([[1, 0, 0, 0, 0],
       [0, 1, 1, 1, 1],
       [1, 1, 1, 0, 0],
       [0, 1, 1, 0, 1],
       [0, 1, 0, 0, 0]]),
np.array([[0, 0, 0, 1, 1],
       [0, 1, 1, 0, 0],
       [0, 0, 1, 0, 1],
       [1, 0, 1, 0, 1],
       [0, 1, 1, 0, 1]]),
np.array([[1, 1, 1, 1, 0],
       [0, 1, 0, 1, 1],
       [0, 0, 1, 0, 1],
       [0, 1, 0, 1, 0],
       [1, 0, 1, 1, 0]]),
np.array([[0, 0, 1, 1, 1],
       [0, 0, 1, 0, 1],
       [0, 1, 0, 0, 1],
       [1, 1, 0, 0, 1],
       [1, 0, 0, 0, 1]]),
np.array([[0, 0, 0, 0, 0],
       [1, 0, 1, 0, 1],
       [1, 1, 1, 0, 0],
       [1, 1, 1, 1, 1],
       [0, 0, 1, 1, 0]]),
np.array([[1, 0, 1, 0, 0],
       [1, 0, 1, 1, 1],
       [0, 0, 0, 1, 1],
       [1, 1, 1, 1, 0],
       [1, 0, 1, 1, 1]]),
np.array([[0, 0, 0, 0, 0],
       [1, 1, 0, 0, 0],
       [1, 1, 1, 0, 1],
       [1, 0, 1, 0, 1],
       [0, 1, 1, 1, 1]]),
np.array([[0, 0, 0, 0, 1],
       [0, 1, 1, 1, 0],
       [1, 1, 0, 1, 1],
       [0, 1, 0, 0, 0],
       [0, 0, 0, 1, 1]]),
np.array([[1, 0, 0, 1, 0],
       [0, 1, 0, 1, 1],
       [0, 0, 1, 0, 0],
       [1, 0, 1, 1, 0],
       [0, 0, 0, 0, 1]]),
np.array([[0, 1, 1, 1, 1],
       [0, 1, 1, 0, 0],
       [1, 0, 0, 0, 0],
       [1, 1, 1, 0, 0],
       [0, 0, 0, 1, 0]]),
np.array([[1, 0, 0, 0, 1],
       [1, 1, 0, 0, 1],
       [0, 1, 0, 0, 0],
       [0, 1, 0, 0, 0],
       [0, 1, 1, 1, 1]]),
np.array([[0, 1, 0, 0, 1],
       [0, 1, 0, 1, 1],
       [1, 1, 1, 0, 1],
       [0, 1, 1, 1, 0],
       [0, 1, 0, 0, 0]]),
np.array([[1, 1, 0, 0, 0],
       [1, 1, 0, 0, 0],
       [1, 1, 0, 0, 0],
       [1, 0, 1, 1, 0],
       [0, 0, 1, 1, 1]]),
np.array([[1, 0, 1, 1, 0],
       [1, 1, 1, 1, 1],
       [1, 0, 0, 1, 0],
       [0, 0, 0, 1, 1],
       [0, 0, 0, 1, 1]]),
np.array([[0, 1, 1, 1, 1],
       [0, 1, 0, 1, 1],
       [1, 1, 0, 1, 1],
       [1, 0, 0, 0, 1],
       [1, 0, 1, 1, 1]]),
np.array([[1, 1, 1, 0, 1],
       [1, 1, 1, 1, 0],
       [0, 0, 0, 1, 1],
       [0, 0, 1, 0, 1],
       [1, 1, 0, 0, 1]]),
np.array([[1, 0, 1, 0, 1],
       [0, 1, 0, 0, 0],
       [0, 1, 1, 0, 0],
       [1, 1, 1, 0, 0],
       [1, 0, 1, 0, 1]]),
np.array([[0, 1, 0, 1, 0],
       [1, 1, 1, 1, 1],
       [1, 0, 1, 0, 0],
       [0, 0, 1, 1, 1],
       [1, 0, 0, 0, 0]]),
np.array([[0, 0, 1, 1, 1],
       [1, 1, 0, 1, 1],
       [0, 0, 1, 0, 1],
       [1, 0, 0, 0, 0],
       [0, 1, 0, 1, 0]]),
np.array([[1, 0, 0, 1, 1],
       [0, 1, 0, 0, 0],
       [0, 1, 0, 0, 1],
       [1, 0, 1, 0, 1],
       [1, 1, 0, 0, 0]]),
np.array([[0, 0, 1, 0, 0],
       [1, 0, 1, 0, 0],
       [0, 0, 1, 0, 1],
       [0, 0, 1, 0, 1],
       [1, 1, 1, 0, 0]]),
np.array([[0, 1, 1, 0, 1],
       [0, 1, 0, 1, 1],
       [0, 0, 0, 0, 0],
       [1, 1, 1, 0, 1],
       [1, 0, 1, 1, 0]]),
np.array([[0, 0, 0, 1, 1],
       [1, 0, 0, 1, 1],
       [0, 1, 1, 0, 0],
       [1, 0, 0, 0, 0],
       [1, 1, 0, 1, 1]]),
np.array([[1, 0, 0, 1, 1],
       [0, 0, 0, 1, 1],
       [0, 0, 1, 1, 0],
       [1, 1, 0, 1, 0],
       [1, 1, 1, 1, 0]]),
np.array([[1, 1, 0, 1, 1],
       [0, 1, 1, 0, 1],
       [0, 0, 0, 1, 1],
       [0, 0, 1, 0, 1],
       [1, 1, 1, 0, 1]]),
np.array([[1, 1, 0, 1, 0],
       [0, 0, 1, 1, 1],
       [1, 0, 1, 0, 1],
       [1, 1, 1, 0, 0],
       [1, 0, 0, 0, 1]]),
np.array([[0, 0, 1, 1, 0],
       [0, 0, 0, 0, 1],
       [1, 1, 0, 1, 0],
       [1, 1, 0, 1, 0],
       [1, 0, 0, 1, 0]]),
np.array([[1, 0, 1, 1, 1],
       [0, 0, 0, 0, 1],
       [1, 1, 1, 1, 1],
       [0, 1, 1, 1, 0],
       [0, 0, 1, 1, 0]]),
np.array([[1, 0, 0, 0, 1],
       [1, 0, 1, 0, 1],
       [1, 1, 0, 1, 1],
       [0, 1, 0, 1, 0],
       [1, 1, 0, 0, 0]]),
np.array([[0, 0, 1, 0, 0],
       [1, 0, 1, 1, 1],
       [1, 1, 0, 0, 0],
       [1, 0, 1, 1, 1],
       [1, 0, 1, 1, 0]]),
np.array([[0, 1, 1, 1, 1],
       [1, 1, 0, 0, 0],
       [1, 1, 1, 0, 1],
       [1, 1, 1, 0, 1],
       [0, 0, 0, 1, 1]]),
np.array([[1, 1, 1, 1, 0],
       [0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0],
       [1, 0, 0, 1, 0],
       [0, 0, 0, 0, 0]]),
np.array([[1, 1, 1, 0, 1],
       [0, 0, 1, 0, 0],
       [1, 1, 0, 1, 1],
       [0, 0, 0, 0, 0],
       [1, 1, 1, 0, 1]]),
np.array([[1, 1, 0, 1, 0],
       [0, 0, 1, 1, 0],
       [0, 0, 0, 0, 0],
       [1, 0, 0, 0, 1],
       [1, 1, 0, 1, 0]]),
np.array([[1, 1, 1, 1, 1],
       [0, 1, 0, 1, 0],
       [0, 0, 0, 0, 1],
       [0, 0, 1, 0, 1],
       [1, 1, 1, 0, 1]]),
np.array([[0, 0, 1, 0, 1],
       [1, 1, 1, 0, 0],
       [1, 0, 1, 1, 1],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 1, 1]]),
np.array([[0, 0, 0, 0, 1],
       [1, 0, 0, 1, 0],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 0],
       [0, 0, 1, 0, 0]]),
np.array([[1, 0, 1, 0, 1],
       [1, 1, 1, 0, 1],
       [0, 0, 0, 1, 0],
       [1, 1, 1, 1, 0],
       [0, 0, 0, 0, 0]]),
np.array([[0, 1, 1, 0, 1],
       [0, 0, 1, 0, 0],
       [1, 1, 1, 1, 1],
       [0, 0, 0, 0, 1],
       [0, 1, 1, 1, 1]]),
np.array([[1, 0, 1, 0, 1],
       [0, 1, 1, 0, 0],
       [1, 0, 1, 1, 0],
       [0, 0, 1, 0, 1],
       [0, 1, 1, 1, 0]])
]

def generate_data():
    np.random.seed(42)
    H = 1
    R = 0
    for i in range(100):
        L = (np.random.rand(5, 5) >= 0.5) * 1
        print(repr(L) + ",")

if __name__ == '__main__':
    generate_data()