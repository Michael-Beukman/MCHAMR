import glob
from typing import Any, Callable, Dict, List, Tuple, Union
import neat
import natsort
import numpy as np
from common.utils import load_compressed_pickle
from contrib.fitness.common.has_patterns_fitness import HasPatternsFitness
from contrib.fitness.common.prob_dist_fitness import ProbabilityDistributionFitness
from contrib.metrics.maze_trivial_solvability_metric import MazeTrivialSolvability
from games.game import Game
from games.level import Level
from metrics.a_star.a_star_metrics import AStarDifficultyMetric, AStarDiversityAndDifficultyMetric, AStarDiversityMetric, AStarEditDistanceDiversityMetric, AStarPathLengthMetric, AStarSolvabilityMetric, AStarWallCountsMetric
from metrics.horn.compression_distance import CompressionDistanceMetric
from metrics.metric import Metric
from metrics.solvability import SolvabilityMetric
from novelty_neat.fitness.fitness import NeatFitnessFunction
from novelty_neat.generation import NeatLevelGenerator

def get_level_with_same_class_different_shape(game: Game, shape: List[int], should_keep_middle_dim_size_if_1 = False) -> Level:
    """Returns a level with the same class as this game's level, but with a different size.

    Args:
        game (Game): The game
        shape (List[int]): new shape, either two numbers for 2D or 3 for 3D

    Returns:
        Level: 
    """
    if should_keep_middle_dim_size_if_1:
        old_shape = game.level.map.shape
        if len(old_shape) == 3 and old_shape[1] == 1:
            shape = (shape[0], 1, shape[-1])
    return game.level.__class__(*shape)

def get_maze_metrics(game: Game, extend=False) -> List[Metric]:
    parent = AStarDiversityAndDifficultyMetric(game, number_of_times_to_do_evaluation=5)
    ans = [SolvabilityMetric(game),
            AStarSolvabilityMetric(game, parent),
            AStarDiversityMetric(game, parent),
            AStarDifficultyMetric(game, parent),
            AStarEditDistanceDiversityMetric(game, parent),]
    
    if extend:
        ans.append(AStarPathLengthMetric(game, parent))
        ans.append(AStarWallCountsMetric(game, parent))
        ans.append(MazeTrivialSolvability(game))
    
    return ans

def _glob_get_first(s: str):
    g = glob.glob(s)
    assert len(g) == 1, f'{s} does not have a length of 1, but of {len(g)}'
    return g[0]

def _glob_get_latest(s: str):
    g = glob.glob(s)
    g = natsort.natsorted(g)
    return g[-1]

def get_pickle_name_from_exp_name(experiment_name: str, seed: int = 0) -> str:
    """Returns a pickle file path from an experiment name (e.g. 2001-a) and a seed

    Args:
        experiment_name (str): 
        seed (int, optional): . Defaults to 0.

    Returns:
        str: fullpath to pickle
    """
    directory = _glob_get_first(f'../results/experiments/pcgnn_{experiment_name}/*/PCGNN/')
    
    directory = _glob_get_latest(f'{directory}/*')
    return  _glob_get_first(f'{directory}/*/*/{seed}/*.pbz2')

def get_generator_net_game_from_pickle(pickle_file: str, shape: List[int] = None, return_entire_dic: bool = False) -> Union[Tuple[NeatLevelGenerator, neat.nn.FeedForwardNetwork, Game], Tuple[NeatLevelGenerator, neat.nn.FeedForwardNetwork, Game, Dict[str, Any]]]:
    """Given a pickle file, returns a level generator, neat network and a game

    Args:
        pickle_file (str): The file to load
        shape (List[int]): New shape of the level, None implies unchanged
        return_entire_dic (bool): If this is true, returns the entire dictionary too, as a fourth return argument. Defaults to False

    Returns:
        Tuple[NeatLevelGenerator, neat.nn.FeedForwardNetwork, Game]:
    """
    dic = load_compressed_pickle(pickle_file)
    generator = dic['extra_results']['entire_method'].level_generator
    game = generator.game
    config = dic['extra_results']['entire_method'].neat_config
    genome = dic['extra_results']['entire_method'].best_agent
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    if shape is not None:
        generator.game.level = get_level_with_same_class_different_shape(game, shape)
    
    if return_entire_dic:
        return generator, net, game, dic
    return generator, net, game


def get_generator_net_game_from_pickle_map_elites(pickle_file: str, shape: List[int] = None, return_entire_dic: bool = False, archive_idx=(0, 0)) -> Union[Tuple[NeatLevelGenerator, neat.nn.FeedForwardNetwork, Game], Tuple[NeatLevelGenerator, neat.nn.FeedForwardNetwork, Game, Dict[str, Any]]]:
    """Given a pickle file, returns a level generator, neat network and a game. This works for map elites.

    Args:
        pickle_file (str): The file to load
        shape (List[int]): New shape of the level, None implies unchanged
        return_entire_dic (bool): If this is true, returns the entire dictionary too, as a fourth return argument. Defaults to False

    Returns:
        Tuple[NeatLevelGenerator, neat.nn.FeedForwardNetwork, Game]:
    """
    generator, net, game, dic = get_generator_net_game_from_pickle(pickle_file, shape, return_entire_dic)
    archive = dic['train_results'][-1]['archive']
    idx = archive.archive[archive_idx[0], archive_idx[1]]
    if idx == -1: return
    genome = archive.indivs[idx].genome
    config = dic['extra_results']['entire_method'].neat_config
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    generator = dic['extra_results']['entire_method'].level_generator
    return generator, net, game, dic

def get_prob_fit_function(dist, level_class=Level) -> Callable[[int, NeatLevelGenerator], NeatFitnessFunction]:
    def prob_fit(number_of_levels_to_generate, level_gen):
        return ProbabilityDistributionFitness(dist, level=level_class(), number_of_levels_to_generate=number_of_levels_to_generate, level_gen=level_gen)
    return prob_fit

def get_pattern_fit_function(patterns: List[Tuple[np.ndarray, float]]) -> Callable[[int, NeatLevelGenerator], NeatFitnessFunction]:
    def pattern_fit(number_of_levels_to_generate, level_gen):
        return HasPatternsFitness(patterns=patterns, number_of_levels_to_generate=number_of_levels_to_generate, level_gen=level_gen)
    return pattern_fit

def get_fitness_with_additional_arguments(fitness_class: Callable[[int, NeatLevelGenerator], NeatFitnessFunction], **kwargs) -> Callable[[int, NeatLevelGenerator], NeatFitnessFunction]:
    def get_fitness(number_of_levels_to_generate, level_gen):
        return fitness_class(number_of_levels_to_generate=number_of_levels_to_generate, level_gen=level_gen, **kwargs)
    return get_fitness