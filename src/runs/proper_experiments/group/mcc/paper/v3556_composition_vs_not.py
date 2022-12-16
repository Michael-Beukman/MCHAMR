import fire
from contrib.fitness.minecraft.twod.composed.flat_town import Minecraft2DFlatTownFitness
from contrib.generation.clean_generation import HiddenChannelsParams, MemoryParams
from games.maze.maze_level import MazeLevel
from games.minecraft.twod.compose.flat_town import Flat2DTownLevel
from novelty_neat.novelty.distance_functions.distance import visual_diversity_normalised
from runs.proper_experiments.group.mcc.michael.mcc_common import run_mcc
from runs.proper_experiments.group.utils.game_utils import get_prob_fit_function


def main(letter):
    """
        This focuses on generating levels while using interesting novelty distance functions
    """
    prob_fit = get_prob_fit_function({
        'empty': 0.5,
        'filled': 0.5,
    }, level_class=MazeLevel)
    NUM = 3556
    NUM_SEEDS = 10

    {
        'aa': lambda LET: run_mcc(NUM, LET, level_class=Flat2DTownLevel, use_one_hot_encoding=False, additional_fitnesses=[Minecraft2DFlatTownFitness], distance_function=visual_diversity_normalised, max_dist=1, pop_size=50, generations=(g:= 150), dimensions=3, NUM_SEEDS=NUM_SEEDS, do_padding_randomly=False ,number_of_random_variables=1, random_perturb_size=0, multiple_iterations=10, USE_RAY=False, all_weights=[0, 0, 1], generator_kwargs=dict(input_center_tile=True, start_level_with_default_tile=True,  progress_default_tile=0), num_levels=5, intra_novelty_neighbours=4),
        
        'be': lambda LET: run_mcc(NUM, LET, level_class=Flat2DTownLevel, use_one_hot_encoding=True, additional_fitnesses=[Minecraft2DFlatTownFitness], distance_function=visual_diversity_normalised, max_dist=1, pop_size=50, generations=(g:= 150), dimensions=3, NUM_SEEDS=NUM_SEEDS, do_padding_randomly=False ,number_of_random_variables=1, random_perturb_size=0, multiple_iterations=5, USE_RAY=False, all_weights=[0, 0, 1], generator_kwargs=dict(input_center_tile=True, start_level_with_default_tile=True,  progress_default_tile=0), num_levels=5, intra_novelty_neighbours=4),
        
    }[letter](letter)


if __name__ == '__main__':
    fire.Fire(main)