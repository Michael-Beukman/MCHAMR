import numpy as np
from contrib.fitness.common.fitness_utils import get_counts_of_array
from games.level import Level
from novelty_neat.fitness.fitness import IndependentNeatFitnessFunction
import skimage.morphology as morph
from contrib.fitness.minecraft.twod.composed import layouts

class Minecraft2DComposedTownRandomLayoutFitness(IndependentNeatFitnessFunction):
    # Fitness for the high-level, composed town
    def __init__(self, index_to_use, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.correct_ans = layouts.DATA[index_to_use]

    def calc_fitness_single_level(self, level: Level) -> float:
        M = level.map
        assert M.shape[1] == 1
        M = M[:, 0, :]
        assert M.shape == self.correct_ans.shape
        return (M == self.correct_ans).mean()

class Minecraft2DFlatTownRandomLayoutFitness(IndependentNeatFitnessFunction):
    # Fitness for the flat town
    def __init__(self, index_to_use, *args, **kwargs):
        super().__init__(*args, **kwargs)
        W = 2
        A = 1
        R = 0
        town = layouts.DATA[index_to_use]
        assert town.shape == (5, 5)
        perfect_house = np.array([
            [W, W, W, W, W],
            [W, A, A, A, W],
            [W, A, A, A, W],
            [W, A, A, A, W],
            [W, W, W, W, W],
        ])
        big = np.zeros((25, 25))
        for i in range(town.shape[0]):
            for j in range(town.shape[-1]):
                if town[i, j] == 1: # This is a house
                    big[i*5:(i+1)*5, j*5:(j+1)*5] = perfect_house.copy()
                else:
                    assert town[i, j] == 0 # is a road
                    big[i*5:(i+1)*5, j*5:(j+1)*5] = np.zeros_like(perfect_house) * R
        self.correct_ans = big

    def calc_fitness_single_level(self, level: Level) -> float:
        M = level.map
        assert M.shape[1] == 1
        M = M[:, 0, :]
        assert M.shape == self.correct_ans.shape
        return (M == self.correct_ans).mean()