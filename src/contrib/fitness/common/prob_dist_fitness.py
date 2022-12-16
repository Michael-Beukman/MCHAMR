from logging import Logger
from typing import Callable, Dict

import numpy as np
from games.level import Level
from novelty_neat.fitness.fitness import IndependentNeatFitnessFunction
from novelty_neat.generation import NeatLevelGenerator
from scipy.spatial import distance

class ProbabilityDistributionFitness(IndependentNeatFitnessFunction):
    def __init__(self, desired_dist: Dict[str, float], 
                 level: Level,
                 prob_dist_func: Callable[[np.ndarray, np.ndarray], float] = distance.jensenshannon,
                 number_of_levels_to_generate: int = 5, 
                 level_gen: NeatLevelGenerator = None, logger: Logger = ...):
        """This is a fitness function that returns a fitness depending on how well the level's distribution of tiles aligns with a given probability distribution.

        Args:
            desired_dist (Dict[str, float]): E.g. {'empty': 0.2, 'filled': 0.8} -- incentivise levels to have 80% wall tiles
            level (Level): A level, just to get the tile dist.
            prob_dist_func (Callable[[np.ndarray, np.ndarray], float], optional): This is a distance function to use to compare two probability distributions. Defaults to distance.jensenshannon.
        """
        super().__init__(number_of_levels_to_generate, level_gen, logger)
        self.level = level
        
        self.desired_dist = desired_dist
        self.keys = sorted(self.desired_dist.keys())
        assert self.keys == sorted(level.tile_types_reversed.keys())
        self.prob_dist_func = prob_dist_func
        self.desired_dist_array = self._convert_dict_to_array_prob_dist(self.desired_dist)

    def calc_fitness_single_level(self, level: Level) -> float:
        # Return 1 - jensenshannon
        dist = self._calc_probability_distribution(level)
        dist = self._convert_dict_to_array_prob_dist(dist)
        return 1 - self.prob_dist_func(dist, self.desired_dist_array)

        
    def _calc_probability_distribution(self, level: Level) -> Dict[str, float]:
        """Returns an unnormalised probability distribution of tiles from this level, e.g. {'empty':50, 'filled':50} represents two tiles having equal number of occurrences in the level.

        Args:
            level (Level): 

        Returns:
            Dict[str, float]: Counts for each tile type.
        """
        this_dist = [0 for _ in self.keys]
        ans = {}
        for tile in np.unique(level.map): # for each tile in the level
            name = level.tile_types[tile]
            this_count = (level.map == tile).sum() # how many tiles have this value
            ans[name] = this_count
        return ans
    
    def _convert_dict_to_array_prob_dist(self, dic: Dict[str, float], normalise=True) -> np.ndarray:
        """Converts a dictionary of tile names to counts into a proper probability distribution, i.e. an array of floats.

        Args:
            dic (Dict[str, float]): 
            normalise (bool, optional): If true, normalises it. Defaults to True.

        Returns:
            np.ndarray: 
        """
        ans = []
        for key in self.keys:
            ans.append(float(dic.get(key, 0))) # 0 if not in dic as we did not see any of those
        
        ans = np.array(ans)
        if normalise: ans /= ans.sum()
        return ans
        