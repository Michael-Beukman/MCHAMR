from typing import Callable, List

import numpy as np
from games.level import Level
from novelty_neat.fitness.fitness import CombinedFitness
from novelty_neat.types import LevelNeuralNet

Schedule = Callable[[int], List[float]]

class Anneal:
    def __init__(self, max_gen,
                 starts: List[float] = [0, 0, 1],
                 ends: List[float]   = [0, 1, 1],
                 change_every_n = 1,
                 pad_steps: int=0) -> None:
        
        self.pad_steps = pad_steps
        assert self.pad_steps * 2 < max_gen

        starts = np.array(starts)
        self.max_gen = max_gen - 2 * pad_steps
        self.starts = starts
        self.ends = ends
        self.diff = np.array(ends) - starts
        self.change_every_n = change_every_n
        self.counter = 0
        self.prev = None
        
    def __call__(self, i):
        self.counter += 1
        if i < self.pad_steps or i >= self.max_gen + self.pad_steps:
            if self.prev is not None: ans = self.prev
            else: ans = self.starts

            print(f"{i} is now in the pad range of {self.pad_steps} -- {self.max_gen} -- val now {ans}")
            return ans
        
        i = i - self.pad_steps
        val: float = i / self.max_gen
        
        if self.counter % self.change_every_n == 0 or i == 0:
            self.counter = 1
            ans = self.starts + val * self.diff
            self.prev = ans
        else:
            ans = self.prev
        print(i, self.counter, self.prev, "ANS = ", ans)
        return ans
        
def schedule_anneal(max_gen,
                    starts: List[float] = [0, 0, 1],
                    ends: List[float]   = [0, 1, 1],
                    *args, **kwargs) -> Schedule:
    return Anneal(max_gen, starts, ends, *args, **kwargs)
    # Returns a schedule where we anneal from the start to the end over the number of generations
    starts = np.array(starts)
    diff = np.array(ends) - starts
    def sched(i):
        val: float = i / max_gen
        return starts + val * diff
    return sched
        
        

class CombinedWeightScheduleFitness(CombinedFitness):
    """This is a combined fitness, but the weights can vary as we go through time.
    """
    def __init__(self, 
                 schedule: Schedule,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.schedule = schedule
        self._count = 0
    
    def calc_fitness(self, nets: List[LevelNeuralNet], levels: List[List[Level]]) -> List[float]:
        weights = self.schedule(self._count)
        self.weights: np.ndarray = np.array(weights) / sum(weights)
        self._count += 1
        return super().calc_fitness(nets, levels)
        