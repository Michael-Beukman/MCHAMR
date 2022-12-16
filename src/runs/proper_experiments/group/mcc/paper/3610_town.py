
import fire
from contrib.fitness.minecraft.twod.town.barcelona_town import Minecraft2DBarcelonaTownFitness
from contrib.generation.clean_generation import MemoryParams
from games.maze.maze_level import MazeLevel
from games.minecraft.twod.town import Minecraft2DTownLevel
from novelty_neat.novelty.distance_functions.distance import visual_diversity_normalised
from runs.proper_experiments.group.mcc.michael.mcc_common import run_mcc
from runs.proper_experiments.group.utils.game_utils import get_prob_fit_function


def main(letter):

    NUM = 3610
    NUM_SEEDS = 4

    {
        'aa': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=None, distance_function=visual_diversity_normalised, max_dist=1, pop_size=30, generations=350, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=False, USE_RAY=False),
        
        'ab': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=None, distance_function=visual_diversity_normalised, max_dist=1, pop_size=30, generations=350, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=False, multiple_iterations=5, USE_RAY=False),
        
        'ac': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=None, distance_function=visual_diversity_normalised, max_dist=1, pop_size=30, generations=350, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=True, USE_RAY=False),
        
        'ad': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=None, distance_function=visual_diversity_normalised, max_dist=1, pop_size=30, generations=350, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=True, multiple_iterations=5, USE_RAY=False),

        'ba': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=[0, 0, 1], distance_function=visual_diversity_normalised, max_dist=1, pop_size=30, generations=350, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=False, USE_RAY=False),
        
        'bb': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=[0, 0, 1], distance_function=visual_diversity_normalised, max_dist=1, pop_size=30, generations=350, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=False, multiple_iterations=5, USE_RAY=False),
        
        'bc': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=[0, 0, 1], distance_function=visual_diversity_normalised, max_dist=1, pop_size=30, generations=350, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=True, USE_RAY=False),
        
        'bd': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=[0, 0, 1], distance_function=visual_diversity_normalised, max_dist=1, pop_size=30, generations=350, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=True, multiple_iterations=5, USE_RAY=False),
        
        'ca': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=[1, 1, 8], distance_function=visual_diversity_normalised, max_dist=1, pop_size=30, generations=350, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=False, USE_RAY=False),
        
        'cb': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=[1, 1, 8], distance_function=visual_diversity_normalised, max_dist=1, pop_size=30, generations=350, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=False, multiple_iterations=5, USE_RAY=False),
        
        'cc': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=[1, 1, 8], distance_function=visual_diversity_normalised, max_dist=1, pop_size=30, generations=350, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=True, USE_RAY=False),
        
        'cd': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=[1, 1, 8], distance_function=visual_diversity_normalised, max_dist=1, pop_size=30, generations=350, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=True, multiple_iterations=5, USE_RAY=False),
        
        
        'baa': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=None, distance_function=visual_diversity_normalised, max_dist=1, pop_size=30, generations=350, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=False, USE_RAY=False, context_size=2),
        
        'bab': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=None, distance_function=visual_diversity_normalised, max_dist=1, pop_size=30, generations=350, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=False, multiple_iterations=5, USE_RAY=False, context_size=2),
        
        'bac': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=None, distance_function=visual_diversity_normalised, max_dist=1, pop_size=30, generations=350, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=True, USE_RAY=False, context_size=2),
        
        'bad': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=None, distance_function=visual_diversity_normalised, max_dist=1, pop_size=30, generations=350, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=True, multiple_iterations=5, USE_RAY=False, context_size=2),

        'bba': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=[0, 0, 1], distance_function=visual_diversity_normalised, max_dist=1, pop_size=30, generations=350, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=False, USE_RAY=False, context_size=2),
        
        'bbb': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=[0, 0, 1], distance_function=visual_diversity_normalised, max_dist=1, pop_size=30, generations=350, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=False, multiple_iterations=5, USE_RAY=False, context_size=2),
        
        'bbc': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=[0, 0, 1], distance_function=visual_diversity_normalised, max_dist=1, pop_size=30, generations=350, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=True, USE_RAY=False, context_size=2),
        
        'bbd': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=[0, 0, 1], distance_function=visual_diversity_normalised, max_dist=1, pop_size=30, generations=350, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=True, multiple_iterations=5, USE_RAY=False, context_size=2),
        
        'bca': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=[1, 1, 8], distance_function=visual_diversity_normalised, max_dist=1, pop_size=30, generations=350, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=False, USE_RAY=False, context_size=2),
        
        'bcb': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=[1, 1, 8], distance_function=visual_diversity_normalised, max_dist=1, pop_size=30, generations=350, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=False, multiple_iterations=5, USE_RAY=False, context_size=2),
        
        'bcc': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=[1, 1, 8], distance_function=visual_diversity_normalised, max_dist=1, pop_size=30, generations=350, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=True, USE_RAY=False, context_size=2),
        
        'bcd': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=[1, 1, 8], distance_function=visual_diversity_normalised, max_dist=1, pop_size=30, generations=350, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=True, multiple_iterations=5, USE_RAY=False, context_size=2),
        
        
        
        'caa': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=[0, 0, 1], distance_function=visual_diversity_normalised, max_dist=1, pop_size=30, generations=1000, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=False, USE_RAY=False),
        'cab': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=[0, 0, 1], distance_function=visual_diversity_normalised, max_dist=1, pop_size=50, generations=1000, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=False, USE_RAY=False),
        'cac': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, additional_fitnesses=[Minecraft2DBarcelonaTownFitness], all_weights=[0, 0, 1], distance_function=visual_diversity_normalised, max_dist=1, pop_size=50, generations=1000, do_padding_randomly=False, number_of_random_variables=1, random_perturb_size=0.1, NUM_SEEDS=NUM_SEEDS, normalisation_range_for_coordinates=(-10, 10), use_one_hot_encoding=False, USE_RAY=False, generator_kwargs=dict(memory_kwargs=MemoryParams(use_memory=True, mem_dim=4))),
    }[letter](letter)


if __name__ == '__main__':
    fire.Fire(main)