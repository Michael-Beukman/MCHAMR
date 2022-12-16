""" This script tries to showcase the usage of the general script, and how that can facilitate generation of levels
"""

import numpy as np
from contrib.fitness.minecraft.house.empty_house_fitness import EmptyHouseFitness
from games.maze.maze_game import MazeGame
from games.minecraft.new.house.minecraft_binary_house_level import MinecraftBinaryHouseLevel
from novelty_neat.general.neat_generate_general_level import GenerateGeneralLevelUsingTiling, GenerateGeneralLevelUsingTiling3D
from novelty_neat.maze.neat_maze_fitness import SolvabilityFitness
from novelty_neat.novelty.distance_functions.distance import euclidean_distance
from novelty_neat.novelty.novelty_metric import NoveltyArchive
from runs.proper_experiments.general_pcgnn_experiment import FitnessParameters, TrainingParameters, general_pcgnn_experiment
from games.maze.maze_level import MazeLevel
from runs.proper_experiments.group.utils.game_utils import get_maze_metrics


def main():
    general_pcgnn_experiment(
        exp_number='3001',          # The number of the experiment -- think of this as the major version
        exp_letter='a',             # The letter of the experiment, similar to a minor version
        LEVEL_CLASS=MinecraftBinaryHouseLevel,      # The class of the level to generate. This controls the levels you generate in the end.
        
        # The level generator: How do we actually generate a level from a neural network.
        # The api requires this to be a function that takes in a game class and returns a LevelGenerator.
        get_level_generator=lambda game: GenerateGeneralLevelUsingTiling3D(
            game,                         # The game
            context_size=1,               # The size of the context, how many tiles surrounding the center one do we give to the network
            number_of_random_variables=2, # How many random variables do we input to the network
            do_padding_randomly=False,    # Is the padding surrounding the level random or not. False means padded with -1
            random_perturb_size=0.1565,   # The size of the random perturbations to all the network's inputs
            predict_size=1,               # How many tiles do we predict at each step. 1 => 1x1 block, 2 => 2x2 block.
            reversed_direction=False,     # Should the iterations go from top left to bottom right or vice versa.
            use_one_hot_encoding=True,    # Should the input tiles be one-hot encoded or just given as integers
            ),
        
        metrics_to_use=[], # What metrics to use to evaluate the levels with -- minecraft has no ones (yet)
        
        # The parameters affecting the fitness
        fitness_params=FitnessParameters(num_levels=15, # How many levels are used for the fitness calc. for each network
                                         num_neighbours_novelty=10, # How many neighbours are used for the novelty calc.
                                         lambd=0,                   # How many indivs. are added to the archive at each generation
                                         archive_mode=NoveltyArchive.RANDOM,    # How are these indivs. added, randomly, or are only the most novel ones added.
                                         intra_novelty_neighbours=10), # How many neighbours are used for the intra-novelty calculation.
        
        # Governing the training parameters
        training_params=TrainingParameters(pop_size=20,                 # How many individuals are in the population
                                           number_of_generations=5),   # How many generations do we run for
        
        additional_fitnesses=[EmptyHouseFitness],                      # What other fitness do we use? Novelty and intra-novelty are already there.
        all_weights=None,                                               # How much to weigh all the fitnesses. None means equal weight for everything, [1, 2, 3] means novelty has a weight of 1, intra-novelty 2 and solvability 3.
        distance_function=euclidean_distance,                           # What distance function to use for novelty.
        max_distance=np.sqrt(10**3),                                                # What is the maximum value of this function? In most cases, try to make the function itself return something between 0 and 1. This is just a catch-all if that is not feasible. sqrt(10^3) is the max Euclidean distance for a binary array of size 10
        num_seeds_to_run=1)                                            # How many seeds to run in parallel. At least 10 for proper experiments, more than 2 will be hard to do locally.

if __name__ == '__main__':
    main()