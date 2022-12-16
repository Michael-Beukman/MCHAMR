import numpy as np
from games.level import Level
from novelty_neat.fitness.fitness import IndependentNeatFitnessFunction
import skimage.morphology as morph


class SimpleHouseRoofFitness(IndependentNeatFitnessFunction):
    def calc_fitness_single_level(self, level: Level) -> float:
        AIR = level.tile_types_reversed['empty']
        WALL = level.tile_types_reversed['brick']
        ROOF = level.tile_types_reversed['roof']
        
        
        a = (level.map[1:-1, 1:-1, 1:-1] == AIR).mean()
        b = (level.map[:, 0, :] == ROOF).mean()
        c = np.mean([
            (level.map[:, -1, :] == WALL).mean(),
            (level.map[0, :, :] == WALL).mean(),
            (level.map[-1, :, :] == WALL).mean(),
            (level.map[:, :, 0] == WALL).mean(),
            (level.map[:, :, -1] == WALL).mean(),
        ])
        
        return (a + b + c) / 3