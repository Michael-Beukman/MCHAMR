""" This script contains all of the common code required to be able to reproduce the evocraft results.
"""

from typing import Tuple, Union
from contrib.fitness.common.combined_fitness_weight_schedule import Schedule
from contrib.generation.clean_generation import GenerateTilingLevel2D, GenerateTilingLevel3D
from novelty_neat.general.neat_generate_general_level import GenerateLevelUsingTilingMultipleIterations
from novelty_neat.novelty.novelty_metric import NoveltyArchive
from runs.proper_experiments.general_pcgnn_experiment import FitnessParameters, TrainingParameters, general_pcgnn_experiment
from games.maze.maze_level import MazeLevel

def run_mcc(number, letter,
            level_class, use_one_hot_encoding, additional_fitnesses,
            all_weights, distance_function, max_dist, pop_size, generations,
            do_padding_randomly, number_of_random_variables, random_perturb_size,
            input_coords=False, normalisation_range_for_coordinates: Tuple[float, float] = (0, 1),
            NUM_SEEDS=10,
            context_size=1,
            predict_size=1,
            multiple_iterations: int = 1,
            num_levels: int=15,
            intra_novelty_neighbours=10,
            USE_RAY=True,
            weight_schedule: Union[Schedule, None] = None,
            dimensions: int = 3,
            generator_kwargs: dict = {},
            metrics_to_use=[],
            generator_class=None,
            num_neighbours_novelty=10):
    
    def get_generator(game):
        class_name = GenerateTilingLevel3D if dimensions == 3 else GenerateTilingLevel2D
        if generator_class is not None:
            def test2():
                def test(*args, **kwargs):
                    return generator_class(dimensions, *args, **kwargs)
                return test
            class_name = test2()
        
        gen = class_name(
            game,                                                   # The game
            context_size=context_size,                                 # The size of the context, how many tiles surrounding the center one do we give to the network
            number_of_random_variables=number_of_random_variables, # How many random variables do we input to the network
            do_padding_randomly=do_padding_randomly,        # Is the padding surrounding the level random or not. False means padded with -1
            random_perturb_size=random_perturb_size,        # The size of the random perturbations to all the network's inputs
            predict_size=predict_size,                                 # How many tiles do we predict at each step. 1 => 1x1 block, 2 => 2x2 block.
            reversed_direction=False,                       # Should the iterations go from top left to bottom right or vice versa.
            use_one_hot_encoding=use_one_hot_encoding,      # Should the input tiles be one-hot encoded or just given as integers
            input_coords=input_coords,
            normalisation_range_for_coordinates=normalisation_range_for_coordinates,
            **generator_kwargs
            )
        if multiple_iterations > 1:
            gen = GenerateLevelUsingTilingMultipleIterations(gen, multiple_iterations)
        return gen
    
    print(f'Running {NUM_SEEDS} seeds')
    general_pcgnn_experiment(
        exp_number=number,          # The number of the experiment -- think of this as the major version
        exp_letter=letter,             # The letter of the experiment, similar to a minor version
        LEVEL_CLASS=level_class,      # The class of the level to generate. This controls the levels you generate in the end.
        
        # The level generator: How do we actually generate a level from a neural network.
        # The api requires this to be a function that takes in a game class and returns a LevelGenerator.
        get_level_generator=get_generator,
        
        metrics_to_use=metrics_to_use, # What metrics to use to evaluate the levels with -- minecraft has no ones (yet)
        
        # The parameters affecting the fitness
        fitness_params=FitnessParameters(num_levels=num_levels, # How many levels are used for the fitness calc. for each network
                                         num_neighbours_novelty=num_neighbours_novelty, # How many neighbours are used for the novelty calc.
                                         lambd=0,                   # How many indivs. are added to the archive at each generation
                                         archive_mode=NoveltyArchive.RANDOM,    # How are these indivs. added, randomly, or are only the most novel ones added.
                                         intra_novelty_neighbours=intra_novelty_neighbours), # How many neighbours are used for the intra-novelty calculation.
        
        # Governing the training parameters
        training_params=TrainingParameters(pop_size=pop_size,                 # How many individuals are in the population
                                           number_of_generations=generations),   # How many generations do we run for
        
        additional_fitnesses=additional_fitnesses,                      # What other fitness do we use? Novelty and intra-novelty are already there.
        all_weights=all_weights,                                               # How much to weigh all the fitnesses. None means equal weight for everything, [1, 2, 3] means novelty has a weight of 1, intra-novelty 2 and solvability 3.
        distance_function=distance_function,                           # What distance function to use for novelty.
        max_distance=max_dist,                                                # What is the maximum value of this function? In most cases, try to make the function itself return something between 0 and 1. This is just a catch-all if that is not feasible. sqrt(10^3) is the max Euclidean distance for a binary array of size 10
        num_seeds_to_run=NUM_SEEDS,
        USE_RAY=USE_RAY,
        weight_schedule=weight_schedule)                                            # How many seeds to run in parallel. At least 10 for proper experiments, more than 2 will be hard to do locally.