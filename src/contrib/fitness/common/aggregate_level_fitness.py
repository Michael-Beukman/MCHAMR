from logging import Logger
from typing import List
import numpy as np
from contrib.fitness.common.fitness_utils import get_counts_of_array
from games.level import Level
from games.minecraft.twod.town import Minecraft2DTownLevel
from novelty_neat.fitness.fitness import IndependentNeatFitnessFunction
import skimage.morphology as morph

from novelty_neat.generation import NeatLevelGenerator


class AggregateLevelFitness(IndependentNeatFitnessFunction):
    """A fitness that aggregates the tiles of a level in chunks, i.e. it makes a 10x10 level from a 20x20 original one by aggregating the 2x2 tiles. Blocks that have the same tile in all 4 tiles will be kept, otherwise they will remain a default tile"""
    def __init__(self, fitnesses: List[IndependentNeatFitnessFunction], weights: List[int], tile_size: int=2, default_tile: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_tile = default_tile
        self.tile_size = tile_size
        self.fitnesses = [f(*args, **kwargs) for f in fitnesses]
        self.weights = weights
        assert len(weights) == len(fitnesses)
    
    def calc_fitness_single_level(self, level: Level) -> float:
        ans = 0
        agg_level = self.get_aggregate_level(level)
        for f, w in zip(self.fitnesses, self.weights):
            ans += f.calc_fitness_single_level(agg_level) * w
        return ans
    
    def get_aggregate_level(self, level: Level) -> Level:
        M = level.map
        shape = M.shape
        og_shape = shape
        if len(shape) == 3 and shape[1] == 1:
            og_shape = (shape[0]// self.tile_size, 1, shape[2]// self.tile_size)
            shape = (shape[0], shape[2])
            M = M[:, 0]
        assert all(s % self.tile_size == 0 for s in shape)
        shape = tuple(s // self.tile_size for s in shape)
        arr = np.zeros(shape, dtype=np.int32)
        
        # get the 2x2 subtiles
        
        # temp = M.reshape(shape[0], shape[1], self.tile_size, self.tile_size).swapaxes(1, 2)#.reshape(shape[0], shape[1], self.tile_size * self.tile_size)
        temp = M.reshape(shape[0], self.tile_size, shape[1], self.tile_size).swapaxes(1, 2).reshape(shape[0], shape[1], self.tile_size * self.tile_size)
        mins = temp.min(axis=-1)
        maxs = temp.max(axis=-1)
        
        is_good = np.isclose(mins, maxs)
        
        arr[is_good] = mins[is_good]
        
        arr[~is_good] = self.default_tile
        
        arr = arr.reshape(og_shape)
        return level.__class__.from_map(arr)

    
    def name(self) -> str:
        s = ",".join([f.name() for f in self.fitnesses])
        w = ",".join([str(w) for w in self.weights])
        return f"AGG[{s}; {w}]"


if __name__ == '__main__':
    a = AggregateLevelFitness([], [], 2, default_tile=9)
    maps = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
    ])
    
    maps = np.array([
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], 
    ])
    maps = maps.reshape(20, 1, 20)
    
    maps = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 2, 2, 0, 0, 0, 0],
        [0, 0, 2, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 4, 4, 0, 0, 0, 0],
        [0, 0, 4, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 6, 6],
        [0, 0, 0, 0, 0, 0, 6, 6],
    ]).reshape(8, 1, 8)
    level = Minecraft2DTownLevel.from_map(maps)
    new_level = a.get_aggregate_level(level)
    print(new_level.map)
    print('--')
    print(level.map)