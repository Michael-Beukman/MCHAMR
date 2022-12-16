from contrib.fitness.common.aggregate_level_fitness import AggregateLevelFitness
from contrib.fitness.minecraft.town.clean_town_fitness import CleanTownEachHouseReachableByRoadsFitness, CleanTownEachHouseReachableByRoadsStricterFitness
import fire
from contrib.fitness.minecraft.twod.town.has_houses_and_gardens import Minecraft2DTownHasHousesAndGardensFitness
from games.maze.maze_level import MazeLevel
from games.minecraft.twod.town import Minecraft2DTownLevel, Minecraft2DTownLevel100Size, Minecraft2DTownLevel20Size, Minecraft2DTownLevel30Size, Minecraft2DTownLevel40Size, Minecraft2DTownLevel50Size
from novelty_neat.novelty.distance_functions.distance import visual_diversity_normalised
from runs.proper_experiments.group.mcc.michael.mcc_common import run_mcc
from runs.proper_experiments.group.utils.game_utils import get_fitness_with_additional_arguments, get_prob_fit_function


def main(letter):
    """
        This focuses on generating levels while using interesting novelty distance functions
    """
    prob_fit = get_prob_fit_function({
        'empty': 0.5,
        'filled': 0.5,
    }, level_class=MazeLevel)
    NUM = 3551
    NUM_SEEDS = 10
    
    G = 0
    R = 1
    H = 2
    good_prob_fit_simple = get_prob_fit_function({'house': 0.4, 'garden': 0.3, 'road': 0.3}, level_class=Minecraft2DTownLevel)
    
    agg_prob_tile1 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[good_prob_fit_simple], weights=[1], tile_size=1, default_tile=1)
    agg_prob_tile2 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[good_prob_fit_simple], weights=[1], tile_size=2, default_tile=1)
    agg_prob_tile3 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[good_prob_fit_simple], weights=[1], tile_size=3, default_tile=1)
    agg_prob_tile4 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[good_prob_fit_simple], weights=[1], tile_size=4, default_tile=1)
    
    agg_prob_tile5 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[good_prob_fit_simple], weights=[1], tile_size=5, default_tile=1)
    agg_prob_tile10 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[good_prob_fit_simple], weights=[1], tile_size=10, default_tile=1)
    
    agg_reach_tile1 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[CleanTownEachHouseReachableByRoadsStricterFitness], weights=[1], tile_size=1, default_tile=1)
    agg_reach_tile2 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[CleanTownEachHouseReachableByRoadsStricterFitness], weights=[1], tile_size=2, default_tile=1)
    agg_reach_tile3 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[CleanTownEachHouseReachableByRoadsStricterFitness], weights=[1], tile_size=3, default_tile=1)
    agg_reach_tile4 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[CleanTownEachHouseReachableByRoadsStricterFitness], weights=[1], tile_size=4, default_tile=1)
    
    agg_reach_tile5 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[CleanTownEachHouseReachableByRoadsStricterFitness], weights=[1], tile_size=5, default_tile=1)
    agg_reach_tile10 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[CleanTownEachHouseReachableByRoadsStricterFitness], weights=[1], tile_size=10, default_tile=1)
    
    agg_counts_tile1 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[Minecraft2DTownHasHousesAndGardensFitness], weights=[1], tile_size=1, default_tile=1)
    agg_counts_tile2 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[Minecraft2DTownHasHousesAndGardensFitness], weights=[1], tile_size=2, default_tile=1)
    agg_counts_tile3 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[Minecraft2DTownHasHousesAndGardensFitness], weights=[1], tile_size=3, default_tile=1)
    agg_counts_tile4 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[Minecraft2DTownHasHousesAndGardensFitness], weights=[1], tile_size=4, default_tile=1)
    
    agg_counts_tile5 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[Minecraft2DTownHasHousesAndGardensFitness], weights=[1], tile_size=5, default_tile=1)
    agg_counts_tile10 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[Minecraft2DTownHasHousesAndGardensFitness], weights=[1], tile_size=10, default_tile=1)



    agg_defaultgrass_prob_tile1 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[good_prob_fit_simple], weights=[1], tile_size=1, default_tile=0)
    agg_defaultgrass_prob_tile2 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[good_prob_fit_simple], weights=[1], tile_size=2, default_tile=0)
    agg_defaultgrass_prob_tile3 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[good_prob_fit_simple], weights=[1], tile_size=3, default_tile=0)
    agg_defaultgrass_prob_tile4 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[good_prob_fit_simple], weights=[1], tile_size=4, default_tile=0)
    agg_defaultgrass_prob_tile5 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[good_prob_fit_simple], weights=[1], tile_size=5, default_tile=0)
    agg_defaultgrass_prob_tile10 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[good_prob_fit_simple], weights=[1], tile_size=10, default_tile=0)
    agg_defaultgrass_reach_tile1 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[CleanTownEachHouseReachableByRoadsStricterFitness], weights=[1], tile_size=1, default_tile=0)
    agg_defaultgrass_reach_tile2 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[CleanTownEachHouseReachableByRoadsStricterFitness], weights=[1], tile_size=2, default_tile=0)
    agg_defaultgrass_reach_tile3 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[CleanTownEachHouseReachableByRoadsStricterFitness], weights=[1], tile_size=3, default_tile=0)
    agg_defaultgrass_reach_tile4 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[CleanTownEachHouseReachableByRoadsStricterFitness], weights=[1], tile_size=4, default_tile=0)
    agg_defaultgrass_reach_tile5 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[CleanTownEachHouseReachableByRoadsStricterFitness], weights=[1], tile_size=5, default_tile=0)
    agg_defaultgrass_reach_tile10 = get_fitness_with_additional_arguments(AggregateLevelFitness, fitnesses=[CleanTownEachHouseReachableByRoadsStricterFitness], weights=[1], tile_size=10, default_tile=0)
    


    {
        'cfa': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, use_one_hot_encoding=False, additional_fitnesses=[good_prob_fit_simple, CleanTownEachHouseReachableByRoadsFitness], distance_function=visual_diversity_normalised, max_dist=1, pop_size=50, generations=(g:= 150), dimensions=3, NUM_SEEDS=NUM_SEEDS, do_padding_randomly=False ,number_of_random_variables=1, random_perturb_size=0, multiple_iterations=10, USE_RAY=False, all_weights=[0, 0,1, 1], generator_kwargs=dict(input_center_tile=True, start_level_with_default_tile=True), num_levels=5, intra_novelty_neighbours=4),
        
        'cfb': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, use_one_hot_encoding=False, additional_fitnesses=[agg_prob_tile1, agg_reach_tile1], distance_function=visual_diversity_normalised, max_dist=1, pop_size=50, generations=(g:= 150), dimensions=3, NUM_SEEDS=NUM_SEEDS, do_padding_randomly=False ,number_of_random_variables=1, random_perturb_size=0, multiple_iterations=10, USE_RAY=False, all_weights=[0, 0,1, 1], generator_kwargs=dict(input_center_tile=True, start_level_with_default_tile=True), num_levels=5, intra_novelty_neighbours=4),
        
        'cfc': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel20Size, use_one_hot_encoding=False, additional_fitnesses=[agg_prob_tile2, agg_reach_tile2], distance_function=visual_diversity_normalised, max_dist=1, pop_size=50, generations=(g:= 150), dimensions=3, NUM_SEEDS=NUM_SEEDS, do_padding_randomly=False ,number_of_random_variables=1, random_perturb_size=0, multiple_iterations=10, USE_RAY=False, all_weights=[0, 0,1, 1], generator_kwargs=dict(input_center_tile=True, start_level_with_default_tile=True), num_levels=5, intra_novelty_neighbours=4),
        
        'cfd': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel30Size, use_one_hot_encoding=False, additional_fitnesses=[agg_prob_tile3, agg_reach_tile3], distance_function=visual_diversity_normalised, max_dist=1, pop_size=50, generations=(g:= 150), dimensions=3, NUM_SEEDS=NUM_SEEDS, do_padding_randomly=False ,number_of_random_variables=1, random_perturb_size=0, multiple_iterations=10, USE_RAY=False, all_weights=[0, 0,1, 1], generator_kwargs=dict(input_center_tile=True, start_level_with_default_tile=True), num_levels=5, intra_novelty_neighbours=4),

        'cfe': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel40Size, use_one_hot_encoding=False, additional_fitnesses=[agg_prob_tile4, agg_reach_tile4], distance_function=visual_diversity_normalised, max_dist=1, pop_size=50, generations=(g:= 150), dimensions=3, NUM_SEEDS=NUM_SEEDS, do_padding_randomly=False ,number_of_random_variables=1, random_perturb_size=0, multiple_iterations=10, USE_RAY=False, all_weights=[0, 0,1, 1], generator_kwargs=dict(input_center_tile=True, start_level_with_default_tile=True), num_levels=5, intra_novelty_neighbours=4),

        'cff': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel50Size, use_one_hot_encoding=False, additional_fitnesses=[agg_prob_tile5, agg_reach_tile5], distance_function=visual_diversity_normalised, max_dist=1, pop_size=50, generations=(g:= 150), dimensions=3, NUM_SEEDS=NUM_SEEDS, do_padding_randomly=False ,number_of_random_variables=1, random_perturb_size=0, multiple_iterations=10, USE_RAY=False, all_weights=[0, 0,1, 1], generator_kwargs=dict(input_center_tile=True, start_level_with_default_tile=True), num_levels=5, intra_novelty_neighbours=4),

        'cfg': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel100Size, use_one_hot_encoding=False, additional_fitnesses=[agg_prob_tile10, agg_reach_tile10], distance_function=visual_diversity_normalised, max_dist=1, pop_size=50, generations=(g:= 150), dimensions=3, NUM_SEEDS=NUM_SEEDS, do_padding_randomly=False ,number_of_random_variables=1, random_perturb_size=0, multiple_iterations=10, USE_RAY=False, all_weights=[0, 0,1, 1], generator_kwargs=dict(input_center_tile=True, start_level_with_default_tile=True), num_levels=5, intra_novelty_neighbours=4),


        
        
        'dfa': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, use_one_hot_encoding=False, additional_fitnesses=[Minecraft2DTownHasHousesAndGardensFitness, CleanTownEachHouseReachableByRoadsFitness], distance_function=visual_diversity_normalised, max_dist=1, pop_size=50, generations=(g:= 150), dimensions=3, NUM_SEEDS=NUM_SEEDS, do_padding_randomly=False ,number_of_random_variables=1, random_perturb_size=0, multiple_iterations=10, USE_RAY=False, all_weights=[0, 0,1, 1], generator_kwargs=dict(input_center_tile=True, start_level_with_default_tile=True), num_levels=5, intra_novelty_neighbours=4),
        
        'dfb': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, use_one_hot_encoding=False, additional_fitnesses=[agg_counts_tile1, agg_reach_tile1], distance_function=visual_diversity_normalised, max_dist=1, pop_size=50, generations=(g:= 150), dimensions=3, NUM_SEEDS=NUM_SEEDS, do_padding_randomly=False ,number_of_random_variables=1, random_perturb_size=0, multiple_iterations=10, USE_RAY=False, all_weights=[0, 0,1, 1], generator_kwargs=dict(input_center_tile=True, start_level_with_default_tile=True), num_levels=5, intra_novelty_neighbours=4),
        
        'dfc': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel20Size, use_one_hot_encoding=False, additional_fitnesses=[agg_counts_tile2, agg_reach_tile2], distance_function=visual_diversity_normalised, max_dist=1, pop_size=50, generations=(g:= 150), dimensions=3, NUM_SEEDS=NUM_SEEDS, do_padding_randomly=False ,number_of_random_variables=1, random_perturb_size=0, multiple_iterations=10, USE_RAY=False, all_weights=[0, 0,1, 1], generator_kwargs=dict(input_center_tile=True, start_level_with_default_tile=True), num_levels=5, intra_novelty_neighbours=4),
        
        'dfd': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel30Size, use_one_hot_encoding=False, additional_fitnesses=[agg_counts_tile3, agg_reach_tile3], distance_function=visual_diversity_normalised, max_dist=1, pop_size=50, generations=(g:= 150), dimensions=3, NUM_SEEDS=NUM_SEEDS, do_padding_randomly=False ,number_of_random_variables=1, random_perturb_size=0, multiple_iterations=10, USE_RAY=False, all_weights=[0, 0,1, 1], generator_kwargs=dict(input_center_tile=True, start_level_with_default_tile=True), num_levels=5, intra_novelty_neighbours=4),

        'dfe': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel40Size, use_one_hot_encoding=False, additional_fitnesses=[agg_counts_tile4, agg_reach_tile4], distance_function=visual_diversity_normalised, max_dist=1, pop_size=50, generations=(g:= 150), dimensions=3, NUM_SEEDS=NUM_SEEDS, do_padding_randomly=False ,number_of_random_variables=1, random_perturb_size=0, multiple_iterations=10, USE_RAY=False, all_weights=[0, 0,1, 1], generator_kwargs=dict(input_center_tile=True, start_level_with_default_tile=True), num_levels=5, intra_novelty_neighbours=4),

        'dff': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel50Size, use_one_hot_encoding=False, additional_fitnesses=[agg_counts_tile5, agg_reach_tile5], distance_function=visual_diversity_normalised, max_dist=1, pop_size=50, generations=(g:= 150), dimensions=3, NUM_SEEDS=NUM_SEEDS, do_padding_randomly=False ,number_of_random_variables=1, random_perturb_size=0, multiple_iterations=10, USE_RAY=False, all_weights=[0, 0,1, 1], generator_kwargs=dict(input_center_tile=True, start_level_with_default_tile=True), num_levels=5, intra_novelty_neighbours=4),
        
        'dfg': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel100Size, use_one_hot_encoding=False, additional_fitnesses=[agg_counts_tile10, agg_reach_tile10], distance_function=visual_diversity_normalised, max_dist=1, pop_size=50, generations=(g:= 150), dimensions=3, NUM_SEEDS=NUM_SEEDS, do_padding_randomly=False ,number_of_random_variables=1, random_perturb_size=0, multiple_iterations=10, USE_RAY=False, all_weights=[0, 0,1, 1], generator_kwargs=dict(input_center_tile=True, start_level_with_default_tile=True), num_levels=5, intra_novelty_neighbours=4),
        
        
        'hlb': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel, use_one_hot_encoding=True, additional_fitnesses=[agg_defaultgrass_prob_tile1, agg_defaultgrass_reach_tile1], distance_function=visual_diversity_normalised, max_dist=1, pop_size=50, generations=(g:= 50), dimensions=3, NUM_SEEDS=NUM_SEEDS, do_padding_randomly=False ,number_of_random_variables=1, random_perturb_size=0, multiple_iterations=3, USE_RAY=False, all_weights=[0, 0,1, 1], generator_kwargs=dict(input_center_tile=True, start_level_with_default_tile=True, progress_default_tile=0), num_levels=5, intra_novelty_neighbours=4),
        
        'hlc': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel20Size, use_one_hot_encoding=True, additional_fitnesses=[agg_defaultgrass_prob_tile2, agg_defaultgrass_reach_tile2], distance_function=visual_diversity_normalised, max_dist=1, pop_size=50, generations=(g:= 50), dimensions=3, NUM_SEEDS=NUM_SEEDS, do_padding_randomly=False ,number_of_random_variables=1, random_perturb_size=0, multiple_iterations=3, USE_RAY=False, all_weights=[0, 0,1, 1], generator_kwargs=dict(input_center_tile=True, start_level_with_default_tile=True, progress_default_tile=0), num_levels=5, intra_novelty_neighbours=4),
        
        'hld': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel30Size, use_one_hot_encoding=True, additional_fitnesses=[agg_defaultgrass_prob_tile3, agg_defaultgrass_reach_tile3], distance_function=visual_diversity_normalised, max_dist=1, pop_size=50, generations=(g:= 50), dimensions=3, NUM_SEEDS=NUM_SEEDS, do_padding_randomly=False ,number_of_random_variables=1, random_perturb_size=0, multiple_iterations=3, USE_RAY=False, all_weights=[0, 0,1, 1], generator_kwargs=dict(input_center_tile=True, start_level_with_default_tile=True, progress_default_tile=0), num_levels=5, intra_novelty_neighbours=4),

        'hle': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel40Size, use_one_hot_encoding=True, additional_fitnesses=[agg_defaultgrass_prob_tile4, agg_defaultgrass_reach_tile4], distance_function=visual_diversity_normalised, max_dist=1, pop_size=50, generations=(g:= 50), dimensions=3, NUM_SEEDS=NUM_SEEDS, do_padding_randomly=False ,number_of_random_variables=1, random_perturb_size=0, multiple_iterations=3, USE_RAY=False, all_weights=[0, 0,1, 1], generator_kwargs=dict(input_center_tile=True, start_level_with_default_tile=True, progress_default_tile=0), num_levels=5, intra_novelty_neighbours=4),
        
        'hlf': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel50Size, use_one_hot_encoding=True, additional_fitnesses=[agg_defaultgrass_prob_tile5, agg_defaultgrass_reach_tile5], distance_function=visual_diversity_normalised, max_dist=1, pop_size=50, generations=(g:= 50), dimensions=3, NUM_SEEDS=NUM_SEEDS, do_padding_randomly=False ,number_of_random_variables=1, random_perturb_size=0, multiple_iterations=3, USE_RAY=False, all_weights=[0, 0,1, 1], generator_kwargs=dict(input_center_tile=True, start_level_with_default_tile=True, progress_default_tile=0), num_levels=5, intra_novelty_neighbours=4),
        
        'hlg': lambda LET: run_mcc(NUM, LET, level_class=Minecraft2DTownLevel100Size, use_one_hot_encoding=True, additional_fitnesses=[agg_defaultgrass_prob_tile10, agg_defaultgrass_reach_tile10], distance_function=visual_diversity_normalised, max_dist=1, pop_size=50, generations=(g:= 50), dimensions=3, NUM_SEEDS=NUM_SEEDS, do_padding_randomly=False ,number_of_random_variables=1, random_perturb_size=0, multiple_iterations=3, USE_RAY=False, all_weights=[0, 0,1, 1], generator_kwargs=dict(input_center_tile=True, start_level_with_default_tile=True, progress_default_tile=0), num_levels=5, intra_novelty_neighbours=4),
    }[letter](letter)


if __name__ == '__main__':
    fire.Fire(main)