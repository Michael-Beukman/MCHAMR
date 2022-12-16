from collections import defaultdict
from typing import List, Tuple, Union
import numpy as np
import scipy
from games.level import Level
import scipy.spatial
from games.maze.maze_level import MazeLevel
from novelty_neat.maze.utils import path_length

from novelty_neat.novelty.distance_functions.distance import visual_diversity_normalised

def get_tiles_from_levels(level: np.ndarray, size: Union[int, Tuple[int, int], Tuple[int, int, int]]) -> List[np.ndarray]:
    """This takes in a level and a size, and returns all overlapping windows of this size by using a sliding window

    Args:
        level (np.ndarray): The level, an array. Only 2D supported for now
        size (Union[int, Tuple[int, int], Tuple[int, int, int]]): The size of the window, either an integer for a square or a tuple for a rectangle.

    Returns:
        List[np.ndarray]: A list of windows of size `size`
    """
    dim = len(level.shape)
    if dim == 3 and level.shape[1] == 1:
        dim = 2
        level = level[:, 0]
    # assert dim == 2
    if type(size) == int:
        size = tuple([size] * dim)
    num_tiles = [d - s + 1 for d, s in zip(level.shape, size)]
    all_tiles = []
    if dim == 2:
        for y in range(num_tiles[0]):
            for x in range(num_tiles[1]):
                little_tile = level[y: y + size[0],
                                    x: x + size[1]]
                all_tiles.append(little_tile)
    else:
        for y in range(num_tiles[0]):
            for x in range(num_tiles[1]):
                for z in range(num_tiles[2]):
                    little_tile = level[y: y + size[0],
                                        x: x + size[1],
                                        z: z + size[2],
                                        ]
                    all_tiles.append(little_tile)
    return all_tiles

def hash_tile(tile: np.ndarray, max_numbers=2) -> int:
    """This takes a tile, a small nD array, and hashes it. If the only possible values were 0 and 1, this pretty much flattens the array and converts the binary number to an integer.

    Args:
        tile (np.ndarray): The array
        max_numbers (int, optional): The amount of possible numbers, e.g. 2 is binary. Defaults to 2.

    Returns:
        int: a hash
    """
    s = tile.flatten()
    ans = 0
    for i, val in enumerate(s):
        ans += val * max_numbers ** i
    return int(ans)

def get_level_tile_patterns_prob_dist(level: np.ndarray, size: int = 2, max_numbers: int = 2, add_eps: float = 0) -> np.ndarray:
    """This basically builds up a probability distribution from the patterns in the level.
            From Simon M. Lucas and Vanessa Volz. 2019. Tile Pattern KL-Divergence for Analysing and Evolving Game Levels. GECCO ’19

    Args:
        level (np.ndarray): The level
        size (int, optional): Size of window. Defaults to 2.
        max_numbers (int, optional): The amount of possible tiles in the level. Defaults to 2.
        add_eps (float, optional): Whether or not to add in an epsilon to each element in the distribution. Defaults to 0.

    Returns:
        np.ndarray: _description_
    """
    tiles = get_tiles_from_levels(level, size)
    hashes = [hash_tile(t) for t in tiles]
    max_number = max_numbers ** (size ** len(level.shape)) # how big is this overall
    temp = np.zeros(max_number, dtype=np.float32)
    for h in hashes:
        temp[h] += 1
    return temp + add_eps


def get_level_tile_patterns_prob_dist_efficient(level: np.ndarray, size: int = 2, max_numbers: int = 2, add_eps: float = 0) -> np.ndarray:
    """This basically builds up a probability distribution from the patterns in the level.
            From Simon M. Lucas and Vanessa Volz. 2019. Tile Pattern KL-Divergence for Analysing and Evolving Game Levels. GECCO ’19
        
        This does so efficiently, using a dictionary instead of a numpy array

    Args:
        level (np.ndarray): The level
        size (int, optional): Size of window. Defaults to 2.
        max_numbers (int, optional): The amount of possible tiles in the level. Defaults to 2.
        add_eps (float, optional): Whether or not to add in an epsilon to each element in the distribution. Defaults to 0.

    Returns:
        np.ndarray: _description_
    """
    tiles = get_tiles_from_levels(level, size)
    hashes = [hash_tile(t) for t in tiles]
    max_number = max_numbers ** (size ** len(level.shape)) # how big is this overall
    temp = defaultdict(lambda: 0)
    for h in hashes:
        temp[h] += 1
    for k in temp: temp[k] += + add_eps
    return temp

def compare_tile_patterns_distance_function_efficiently(size: int, max_numbers: int, a: np.ndarray, b: np.ndarray) -> float:
    """Returns the jensen shannon divergence between the probability distributions over tile patterns from different levels.

    Args:
         size (int, optional): Size of window.
        max_numbers (int, optional): The amount of possible tiles in the level.
        a (np.ndarray): 
        b (np.ndarray): 

    Returns:
        float: dist
    """
    d_a = get_level_tile_patterns_prob_dist_efficient(a, size, max_numbers, add_eps=1e-4)
    d_b = get_level_tile_patterns_prob_dist_efficient(b, size, max_numbers, add_eps=1e-4)
    keys = set(d_a.keys()) | set(d_b.keys())
    
    p_a = np.zeros(len(keys), dtype=np.float32)
    p_b = np.zeros(len(keys), dtype=np.float32)
    for i, k in enumerate(keys):
        p_a[i] = d_a.get(k, 1e-4)
        p_b[i] = d_b.get(k, 1e-4)

    return np.clip(scipy.spatial.distance.jensenshannon(p_a, p_b, base=2), 0, 1)



def compare_tile_patterns_distance_function(size: int, max_numbers: int, a: np.ndarray, b: np.ndarray) -> float:
    """Returns the jensen shannon divergence between the probability distributions over tile patterns from different levels.

    Args:
         size (int, optional): Size of window.
        max_numbers (int, optional): The amount of possible tiles in the level.
        a (np.ndarray): 
        b (np.ndarray): 

    Returns:
        float: dist
    """
    p_a = get_level_tile_patterns_prob_dist(a, size, max_numbers)
    p_b = get_level_tile_patterns_prob_dist(b, size, max_numbers)

    return np.clip(scipy.spatial.distance.jensenshannon(p_a, p_b, base=2), 0, 1)

def compare_tile_patterns_distance_functions_range(size_min: int, size_max: int, max_numbers: int, *args) -> float:
    """This returns the average distance (`compare_tile_patterns_distance_function`) over a range of tile sizes.

    Args:
        size_min (int): Min size, inclusive
        size_max (int): Max size, inclusive
        max_numbers (int): The amount of possible tiles in the level.

    Returns:
        float: distance
    """
    ans = 0
    count = 0
    for i in range(size_min, size_max+1):
        count += 1
        ans += compare_tile_patterns_distance_function(i, max_numbers, *args)
    return np.clip(ans / count, 0, 1)

def compare_tile_patterns_distance_functions_range_1_3_2(*args):
    return compare_tile_patterns_distance_functions_range(1, 3, 2, *args)

def compare_tile_patterns_distance_functions_range_2_3_2(*args):
    return compare_tile_patterns_distance_functions_range(2, 3, 2, *args)

def compare_tile_patterns_distance_functions_range_2_4_2(*args):
    return compare_tile_patterns_distance_functions_range(2, 4, 2, *args)

def compare_tile_patterns_distance_functions_range_3_6_2(*args):
    return compare_tile_patterns_distance_functions_range(3, 6, 2, *args)

def compare_tile_patterns_distance_function_1_2(*args):
    return compare_tile_patterns_distance_function(1, 2, *args)

def compare_tile_patterns_distance_function_3_2(*args):
    return compare_tile_patterns_distance_function(3, 2, *args)

def compare_tile_patterns_distance_function_4_2(*args):
    return compare_tile_patterns_distance_function(4, 2, *args)

def compare_tile_patterns_distance_function_5_2(*args):
    return compare_tile_patterns_distance_function(5, 2, *args)

def compare_tile_patterns_distance_function_6_2(*args):
    return compare_tile_patterns_distance_function(6, 2, *args)

def compare_tile_patterns_distance_function_2_2(*args):
    return compare_tile_patterns_distance_function(2, 2, *args)

def compare_tile_patterns_distance_function_2_3(*args):
    return compare_tile_patterns_distance_function(2, 3, *args)

def compare_tile_patterns_distance_function_2_6(*args):
    return compare_tile_patterns_distance_function(2, 6, *args)


def compare_tile_patterns_distance_function_efficiently_1_2(*args):
    return compare_tile_patterns_distance_function_efficiently(1, 2, *args)

def compare_tile_patterns_distance_function_efficiently_3_2(*args):
    return compare_tile_patterns_distance_function_efficiently(3, 2, *args)

def compare_tile_patterns_distance_function_efficiently_4_2(*args):
    return compare_tile_patterns_distance_function_efficiently(4, 2, *args)

def compare_tile_patterns_distance_function_efficiently_5_2(*args):
    return compare_tile_patterns_distance_function_efficiently(5, 2, *args)

def compare_tile_patterns_distance_function_efficiently_6_2(*args):
    return compare_tile_patterns_distance_function_efficiently(6, 2, *args)

def compare_tile_patterns_distance_function_efficiently_2_2(*args):
    return compare_tile_patterns_distance_function_efficiently(2, 2, *args)

def compare_tile_patterns_distance_function_efficiently_2_3(*args):
    return compare_tile_patterns_distance_function_efficiently(2, 3, *args)


def compare_tile_patterns_distance_function_efficiently_2_5(*args):
    return compare_tile_patterns_distance_function_efficiently(2, 5, *args)

def compare_tile_patterns_distance_function_efficiently_2_6(*args):
    return compare_tile_patterns_distance_function_efficiently(2, 6, *args)


# Matthew's Functions
def get_correlaton_coefs(curr_window_a : np.ndarray, b_history: np.ndarray,  window_size : int, end_point : int, coer_thresh = 0.2):
    """
    
    """
    alls = []
    for i in range(window_size+1, end_point):
        b_window = b_history[:, i-window_size-1 : i]
        coer_vals = (np.corrcoef(curr_window_a, b_window))
        for coer_val in coer_vals.flatten(): 
            if (coer_val > coer_thresh) and not np.isnan(coer_val):
                alls.append(coer_val)
    return alls

def compare_levels_corr_coeff(a: np.ndarray, b: np.ndarray, min_window_size : int = 5, max_window_size : int = 10, coer_thresh = 0.2) -> float:
    """
    
    """
    coer_vals = []
    if a.shape != b.shape:
        print("A and B must be the same shape")
        raise Exception
    for curr_window_size in range(min_window_size, max_window_size): # for each window size
        base = 0
        for i in range(curr_window_size + 1, a.shape[1]):  # for each possible snippet
            curr_a_window = a[:,base:i]
            b_history = b[:, :i]
            temp = get_correlaton_coefs(curr_window_a=curr_a_window, b_history=b_history, window_size=curr_window_size, end_point = i, coer_thresh=coer_thresh)
            for t in temp:
                coer_vals.append(t)
            base += 1
    coer_vals.sort()

    if (len(coer_vals)) == 0:
        return 0
    
    highest_coers = coer_vals[-20:]
    return np.clip(1 - sum(highest_coers)/len(highest_coers) , 0, 1)

# Muhammad's function:

def hausdorff_distance(a: np.ndarray, b: np.ndarray):

    dist,_x,_y = scipy.spatial.distance.directed_hausdorff(a,b,seed=72)
    return dist


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten()
    b = b.flatten()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0
    ans = np.clip(a.dot(b)/ (na*nb), 0, 1)
    if np.isnan(ans): return 0
    return ans


def visual_diversity_ignore_boundaries(a, b):
    # Ignores boundaries when doing visual diversity.
    if a.shape[1] == 1: a = a[:, 0]
    if b.shape[1] == 1: b = b[:, 0]
    return visual_diversity_normalised(
        a[1:-1, 1:-1],
        b[1:-1, 1:-1],
    )
    
    
    
def maze_path_length_diversity(a, b):
    area = a.size
    a = MazeLevel.from_map(a)
    b = MazeLevel.from_map(b)
    la = path_length(a)
    lb = path_length(b)
    
    return abs(la - lb) / area


def maze_path_length_diversity_div_by_50(a, b):
    area = a.size
    a = MazeLevel.from_map(a)
    b = MazeLevel.from_map(b)
    la = path_length(a)
    lb = path_length(b)
    
    return np.clip(abs(la - lb) / 50, 0, 1)

def maze_number_of_walls_dist(a, b):
    sa = (a == 1).sum()
    sb = (b == 1).sum()
    area = a.size
    
    return np.clip(abs(sa - sb) / area, 0, 1)