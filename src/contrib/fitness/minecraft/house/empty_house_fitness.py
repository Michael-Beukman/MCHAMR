import numpy as np
from games.level import Level
from novelty_neat.fitness.fitness import IndependentNeatFitnessFunction


class EmptyHouseFitness(IndependentNeatFitnessFunction):
    """Returns the fitness of a house as basically how much it overlaps with a one tile thick cube
    """
    def calc_fitness_single_level(self, level: Level) -> float:
        desired_walls = np.zeros_like(level.map)
        for k in [0, -1]:
            desired_walls[k, :, :] = 1
            desired_walls[:, :, k] = 1
            desired_walls[:, k, :] = 1
        TEST = desired_walls.sum()
        how_much_walls = np.clip(((desired_walls == level.map)[desired_walls == 1].sum()) / (TEST - 3), 0, 1)
        
        how_much_air_inside = (1 - level.map[1:-1, 1:-1, 1:-1]).mean()
        A = (how_much_walls + how_much_air_inside) / 2
        return A
