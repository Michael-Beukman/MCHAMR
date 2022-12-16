"""
    This file is tasked with performing training for a general experiment. It aims to keep all of the boilerplate in this file, so running new experiments is super simple and straightforward.
"""
import multiprocessing
import os
from typing import Any, Callable, Dict, List, Tuple, Union
import ray
import neat
import wandb
from common.types import Verbosity
from common.utils import get_date
from experiments.config import Config
from experiments.experiment import Experiment
from games.game import Game
from games.level import Level
from novelty_neat.generation import NeatLevelGenerator
from metrics.metric import Metric
from novelty_neat.fitness.fitness import CombinedFitness, NeatFitnessFunction
from novelty_neat.novelty_neat import NoveltyNeatPCG
from novelty_neat.novelty.distance_functions.distance import NoveltyDistanceFunction, visual_diversity_normalised
from novelty_neat.novelty.novelty_metric import NoveltyArchive, NoveltyIntraGenerator, NoveltyMetric
from contrib.fitness.common.combined_fitness_weight_schedule import Schedule, CombinedWeightScheduleFitness
class _Parameters:
    def __init__(self) -> None:
        self._dict = {}
        pass

    def to_dict(self) -> Dict[str, Any]:
        return self._dict
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        super().__setattr__(__name, __value)
        if __name == '_dict': return
        if '_' not in __name[:1]: self._dict[__name] = __value
    
class GenerationParameters(_Parameters):
    def __init__(self) -> None:
        pass

class FitnessParameters(_Parameters):
    def __init__(self,
                    num_levels: int = 15, 
                    num_neighbours_novelty: int = 10,
                    lambd: int = 0,
                    archive_mode: NoveltyArchive = NoveltyArchive.RANDOM,
                    intra_novelty_neighbours: int = 10, 
                 ) -> None:
        """
        Args:
            num_levels (int, optional):  How many levels are used for the fitness calc. for each network. Defaults to 15.
            num_neighbours_novelty (int, optional): How many neighbours are used for the novelty calc.. Defaults to 10.
            lambd (int, optional): How many indivs. are added to the archive at each generation. Defaults to 0.
            archive_mode (NoveltyArchive, optional): How are these indivs. added, randomly, or are only the most novel ones added.. Defaults to NoveltyArchive.RANDOM.
            intra_novelty_neighbours (int, optional):  How many neighbours are used for the intra-novelty calculation.. Defaults to 10.
        """
        super().__init__()
        self.num_levels = num_levels
        self.num_neighbours_novelty = num_neighbours_novelty
        self.lambd = lambd
        self.archive_mode = archive_mode
        self.intra_novelty_neighbours = intra_novelty_neighbours

def conditional_decorator(dec, condition):
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)
    return decorator
class TrainingParameters(_Parameters):
    def __init__(self, pop_size: int = 50, number_of_generations: int = 50) -> None:
        """
        Args:
            pop_size (int, optional): How many individuals are in the population. Defaults to 50.
            number_of_generations (int, optional):  How many generations do we run for. Defaults to 50.
        """
        super().__init__()
        self.pop_size = pop_size
        self.number_of_generations = number_of_generations

def general_pcgnn_experiment(
    exp_number: str, 
    exp_letter: str,
    LEVEL_CLASS: Level,
    get_level_generator: Callable[[Game], NeatLevelGenerator],
    fitness_params: FitnessParameters,
    training_params: TrainingParameters,
    additional_fitnesses: List[Callable[[int, NeatLevelGenerator], NeatFitnessFunction]] = [],
    metrics_to_use: List[Metric] = [],
    all_weights: List[float] = None,
    distance_function: NoveltyDistanceFunction = visual_diversity_normalised,
    max_distance: float = 1,
    num_seeds_to_run: int = 10,
    wandb_project: str = 'PCGNN2',
    level_size: Tuple[int, int] = None,
    USE_RAY = True,
    weight_schedule: Union[Schedule, None] = None
):

    """This runs a single training run for PCGNN. It pretty much initialises the population, runs them for a set number of generations. It further then saves the results and optionally runs metrics. See src/runs/proper_experiments/group/pcgnn/demo/demo_script.py for usage.

    Args:
        exp_number (str):  The number of the experiment -- think of this as the major version
        exp_letter (str):  The letter of the experiment, similar to a minor version
        
        LEVEL_CLASS (Level): The class of the level to generate. This controls the levels you generate in the end. Note, this is the class and not an object. So, MazeLevel is correct, but MazeLevel() is not
        get_level_generator (Callable[[Game], NeatLevelGenerator]): This is a function, which takes in a Game and returns a NeatLevelGenerator.
        
        fitness_params (FitnessParameters): The parameters that control the fitness calculations
        
        training_params (TrainingParameters): Parameters that control aspects of the training
        
        additional_fitnesses (List[Callable[[int, NeatLevelGenerator], NeatFitnessFunction]], optional):  A list of additional fitness functions. Each is an instance of the `NeatFitnessFunction` class. Defaults to [].
        
        metrics_to_use (List[Metric], optional): Metrics to evaluate the levels with. Defaults to [].
        
        all_weights (List[float], optional): If None, defaults to using an equal weight for each fitness. If not None, then each fitness will have that weight. The first two fitnesses are Novelty and IntraNovelty, the rest are specified by the parameter . Defaults to None.
        
        distance_function (NoveltyDistanceFunction, optional): A distance function for novelty, which takes in two numpy arrays, representing the level and returns a float -- how different they are, normalised to between 0 and 1. Defaults to visual_diversity_normalised.
        
        max_distance (float, optional): The maximum value for the novelty function. Any value other than 1 is discouraged, rather ensure the distance function itself returns a normalised value if at all possible. Defaults to 1.
        
        num_seeds_to_run (int, optional): The number of seeds to run in parallel to see how variable results are across seeds. Would recommend no less than 10. Defaults to 10.
        
        wandb_project (str, optional): The wandb project to save this to. Defaults to 'PCGNN2'.
        
        level_size (Tuple[int, int], optional): The size of the desired levels.

    Returns:
        _type_: _description_
    """   
    if not USE_RAY:
        global single_func
    def _get_level():
        if level_size is not None:
            return (LEVEL_CLASS(*level_size))
        return (LEVEL_CLASS())
    assert training_params.pop_size -1 >= fitness_params.num_neighbours_novelty
    if USE_RAY:
        print("Just before init")
        ray.init(include_dashboard=False, num_cpus=num_seeds_to_run)
        print("Just after init")
    name = f'pcgnn_{exp_number}-{exp_letter}'
    method = 'PCGNN'
    date = get_date()
    # Create our config file
    game = Game(_get_level())
    game_name = game.level.name()
    
    # Where to save the results
    results_directory = f'../results/experiments/{name}/{game_name}/{method}/{date}/{training_params.pop_size}/{training_params.number_of_generations}'
    # A Game
    level_generator = get_level_generator(game)
    
    in_dim, out_dim = level_generator.get_input_output_size()
    config_file = save_config_file(in_dim=in_dim, out_dim=out_dim, pop_size=training_params.pop_size)
    
    # Getting the fitness
    def get_overall_fitness(num_levels: int, num_neighbours_novelty: int, lambd: int,
                            intra_novelty_neighbours: int, archive_mode: NoveltyArchive) -> NeatFitnessFunction:
        funcs = [NoveltyMetric(level_generator, distance_function, max_dist=max_distance, number_of_levels=num_levels, 
                               number_of_neighbours=num_neighbours_novelty, lambd=lambd, archive_mode=archive_mode,should_use_all_pairs=False),
                NoveltyIntraGenerator(num_levels, level_generator, distance_function, max_dist=max_distance, 
                                    number_of_neighbours=min(intra_novelty_neighbours, num_levels - 1))
                ]
        weights = [1, 1]
        if len(additional_fitnesses):
            funcs += [
                a(number_of_levels_to_generate=num_levels,
                  level_gen=level_generator) for a in additional_fitnesses
            ]
            weights += [1 for _ in range(len(additional_fitnesses))] 

        if all_weights is not None:
            assert len(weights) == len(all_weights)
            weights = all_weights
        
        kwargs = dict(fitnesses=funcs, weights=weights, number_of_levels_to_generate=num_levels, 
                               level_gen=level_generator, mode='add')
        class_name = CombinedFitness
        
        if weight_schedule is not None:
            class_name = CombinedWeightScheduleFitness
            kwargs['schedule'] = weight_schedule
        
        return class_name(**kwargs)
    args = {
        'population_size': training_params.pop_size,
        'number_of_generations': training_params.number_of_generations,
        'fitness': get_overall_fitness(**fitness_params.to_dict()).params(),
        'level_gen': 'tiling',
        'config_filename': config_file
    }
    print(args)
    print(f"Running Experiment {name} now")
    print("=" * 100)
    def get_neat_config():
        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           config_file)
    # Getting the population
    def get_pop():
        level = _get_level()
        game = Game(level)
        fitness = get_overall_fitness(**fitness_params.to_dict())
        return NoveltyNeatPCG(game, level, level_generator=level_generator, fitness_calculator=fitness, neat_config=get_neat_config(), num_generations=training_params.number_of_generations)
    # @ray.remote(num_gpus=1.0/num_seeds_to_run)

    @conditional_decorator(ray.remote, USE_RAY)
    def single_func(seed):
        config = Config(
            name=name, game=game_name, method=method,
            seed=seed, results_directory=os.path.join(results_directory, str(seed)),
            method_parameters=args, date=date,
            project=wandb_project
        )
        print("Date = ", config.date, config.results_directory, config.hash(seed=False))
        experiment = Experiment(config, get_pop, metrics_to_use, log_to_wandb=True, verbose=Verbosity.PROGRESS if seed == 0 else Verbosity.NONE)

        experiment.do_all()
        wandb.finish()

        return f"Completed with seed = {seed}"
    
    if USE_RAY:
        futures = [single_func.remote(i) for i in range(num_seeds_to_run)]
        print(ray.get(futures))
    else:
        pool = multiprocessing.Pool(processes=num_seeds_to_run)
        print(pool.map(single_func, range(num_seeds_to_run)))
        pool.close()
        pool.join()



def save_config_file(in_dim, out_dim, pop_size):
    directory = 'runs/proper_experiments/pcgnn2/config/'
    os.makedirs(directory, exist_ok=True)
    fname = f'{directory}/tiling_generate_{in_dim}_{out_dim}_balanced_pop{pop_size}'
    s = f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = 10000000
pop_size              = {pop_size}
reset_on_extinction   = False

[DefaultGenome]
# node activation options
activation_default      = random
activation_mutate_rate  = 0.5
activation_options      = sigmoid sin gauss

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# connection add/remove rates
conn_add_prob           = 0.6
conn_delete_prob        = 0.3

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.1

feed_forward            = True
initial_connection      = full

# node add/remove rates
node_add_prob           = 0.6
node_delete_prob        = 0.3

# network parameters
num_hidden              = 0
num_inputs              = {in_dim}
num_outputs             = {out_dim}

# node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""

    if not os.path.exists(fname):
        with open(fname, 'w+') as f:
            f.write(s)
    return fname

