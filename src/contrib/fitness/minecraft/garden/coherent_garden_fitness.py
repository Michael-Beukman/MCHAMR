from typing import List
import numpy as np
from games.level import Level
from novelty_neat.fitness.fitness import IndependentNeatFitnessFunction


class CoherentGardenFitness(IndependentNeatFitnessFunction):
    """A garden fitness -- not really sure what this does.
    """
    def get_tree_tiles(self, level : Level, tree_ids : List, min_dist : int= 5,):
        # flat two_d 
        tree_locations = np.argwhere(np.logical_and(level.map >= 1, level.map <= 3))
        bad_trees = 0
        for i, tree in enumerate(tree_locations):
            for j, other_tree in enumerate(tree_locations):
                if i != j:
                    if np.linalg.norm(tree - other_tree) <  min_dist:
                        bad_trees += 1
                        break
                    
        return bad_trees


    def calc_fitness_single_level(self, level: Level) -> float:
        min_prop_empty = 0.2 # how much free space is allowed at the minimum
        max_prop_full = 0.7 # how much of the space can be populated at maximum
        water_dampness = 0.05
        dirt, water = 0, 4
        
        sapling_types = [1] # set of ids to represent tree sapling types
        flower_types = [2, 3] # sset of idss to represent flower types
        num_trees = np.logical_and(level.map >= 1, level.map <= 3).sum() 
        num_flowers = np.logical_and(level.map >= 4, level.map <= 5).sum()
        # pretend dirt is blank space
        density_score = 0
        num_dirt = np.mean(level.map == dirt)
        if num_dirt > min_prop_empty and num_dirt < max_prop_full:
            density_score = 1
        num_bad_trees = self.get_tree_tiles(level, sapling_types, min_dist=6)
        tree_score = 0.2
        if num_trees != 0:
            tree_score = 1 - 10*(num_bad_trees/num_trees)
            if tree_score < 0:
                tree_score = 0
        # then have some max proportion that is allowed to be populated e.g not dirt
        water_prop = np.mean(level.map == water)
        water_score = 0
        if water_prop < water_dampness:
            water_score = 1
        has_tree = 0
        has_flowers = 0
        if num_flowers > 0:
            has_flowers = 1
        if num_trees > 0:
            has_tree = 1

        return (density_score + tree_score + water_score + has_tree + has_flowers) / 5
        