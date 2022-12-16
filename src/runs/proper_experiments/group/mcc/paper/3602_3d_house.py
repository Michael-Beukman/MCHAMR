
import fire
from contrib.fitness.minecraft.house.simple_house_roof_fitness import SimpleHouseRoofFitness
from contrib.generation.clean_generation import MemoryParams
from games.maze.maze_level import MazeLevel
from games.minecraft.new.house.minecraft_house_with_roof_level import MinecraftHouseWithRoofLevel
from novelty_neat.novelty.distance_functions.distance import visual_diversity_normalised
from runs.proper_experiments.group.mcc.michael.mcc_common import run_mcc
from runs.proper_experiments.group.utils.game_utils import get_prob_fit_function


def main(letter):

    NUM = 3602
    NUM_SEEDS = 4

    {
        'aaa': lambda LET: run_mcc(NUM, LET, level_class=MinecraftHouseWithRoofLevel, use_one_hot_encoding=False, additional_fitnesses=[SimpleHouseRoofFitness], distance_function=visual_diversity_normalised, max_dist=1, pop_size=20, generations=(g:= 500), dimensions=3, NUM_SEEDS=NUM_SEEDS, do_padding_randomly=False ,number_of_random_variables=1, random_perturb_size=0, multiple_iterations=1, USE_RAY=False,  generator_kwargs=dict(input_center_tile=True), all_weights=[1, 1, 8]),
        
        'aab': lambda LET: run_mcc(NUM, LET, level_class=MinecraftHouseWithRoofLevel, use_one_hot_encoding=False, additional_fitnesses=[SimpleHouseRoofFitness], distance_function=visual_diversity_normalised, max_dist=1, pop_size=20, generations=(g:= 500), dimensions=3, NUM_SEEDS=NUM_SEEDS, do_padding_randomly=False ,number_of_random_variables=1, random_perturb_size=0, multiple_iterations=10, USE_RAY=False,  generator_kwargs=dict(input_center_tile=True), all_weights=[1, 1, 8]),
    }[letter](letter)


if __name__ == '__main__':
    fire.Fire(main)