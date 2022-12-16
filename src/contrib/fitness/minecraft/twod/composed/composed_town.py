import numpy as np
from contrib.fitness.common.fitness_utils import get_counts_of_array
from games.level import Level
from novelty_neat.fitness.fitness import IndependentNeatFitnessFunction
import skimage.morphology as morph

class Minecraft2DComposedTownFitness(IndependentNeatFitnessFunction):
    # Fitness for the high-level, composed town
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        H = 1
        R = 0
        self.correct_ans = np.array([
            [H, R, H, R, H],
            [R, R, H, R, R],
            [H, R, R, R, H],
            [R, R, H, R, R],
            [H, R, H, R, H],
        ])

    def calc_fitness_single_level(self, level: Level) -> float:
        M = level.map
        assert M.shape[1] == 1
        M = M[:, 0, :]
        assert M.shape == self.correct_ans.shape
        return (M == self.correct_ans).mean()