import numpy as np
from contrib.fitness.common.fitness_utils import get_counts_of_array
from games.level import Level
from novelty_neat.fitness.fitness import IndependentNeatFitnessFunction
import skimage.morphology as morph

class Minecraft2DBarcelonaTownFitness(IndependentNeatFitnessFunction):
    def calc_fitness_single_level(self, level: Level) -> float:
        M = level.map
        assert M.shape[1] == 1, f"Incorrect Shape: {M.shape}"
        M = M[:, 0, :]
        # See if all 
        ROAD  = level.tile_types_reversed['road']
        HOUSE = level.tile_types_reversed['house']
        GARDEN = level.tile_types_reversed['garden']
        
        A = (M[2:-2, 2:-2] == GARDEN).mean()
        B = np.mean([
            (M[1:-1, 1] == ROAD).mean(),
            (M[1:-1, -2] == ROAD).mean(), 
            (M[1, 1:-1] == ROAD).mean(),
            (M[-2, 1:-1] == ROAD).mean(), 
        ])
        C = np.mean([
            (M[:, 0]  == HOUSE).mean(),
            (M[:, -1] == HOUSE).mean(), 
            (M[0, :]  == HOUSE).mean(),
            (M[-1, :] == HOUSE).mean(), 
        ])
        return (A+B+C)/3
    