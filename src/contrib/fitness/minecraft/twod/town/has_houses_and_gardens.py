import numpy as np
from contrib.fitness.common.fitness_utils import get_counts_of_array
from games.level import Level
from novelty_neat.fitness.fitness import IndependentNeatFitnessFunction
import skimage.morphology as morph

class Minecraft2DTownHasHousesAndGardensFitness(IndependentNeatFitnessFunction):
    # Does the town have roughly equal numbers of roads gardens and roads.

    def calc_fitness_single_level(self, level: Level) -> float:
        M = level.map
        assert M.shape[1] == 1, f"Incorrect Shape: {M.shape}"
        M = M[:, 0, :]
        # See if all 
        ROAD   = level.tile_types_reversed['road']
        HOUSE  = level.tile_types_reversed['house']
        GARDEN = level.tile_types_reversed['garden']
        
        
        r = (M == ROAD).mean()
        g = (M == GARDEN).mean()
        h = (M == HOUSE).mean()
        
        
        a = np.clip(1 - 10 * (r - 1 / 3) ** 2, 0, 1)
        b = np.clip(1 - 10 * (g - 1 / 3) ** 2, 0, 1)
        c = np.clip(1 - 10 * (h - 1 / 3) ** 2, 0, 1)
        
        return (a + b + c) / 3
    