import fire
from contrib.fitness.minecraft.twod.composed.random_layouts_fitness import Minecraft2DFlatTownRandomLayoutFitness
from games.maze.maze_level import MazeLevel
from games.minecraft.twod.compose.flat_town import Flat2DTownLevel
from novelty_neat.novelty.distance_functions.distance import visual_diversity_normalised
from runs.proper_experiments.group.mcc.michael.mcc_common import run_mcc
from runs.proper_experiments.group.utils.game_utils import get_fitness_with_additional_arguments, get_prob_fit_function

def _get_letter(a, b):
    return chr(a + 97) + chr(b + 97)


def main(letter_to_use):
    """
        This focuses on generating levels while using interesting novelty distance functions
    """
    prob_fit = get_prob_fit_function({
        'empty': 0.5,
        'filled': 0.5,
    }, level_class=MazeLevel)
    NUM = 3560
    NUM_SEEDS = 10

    ALL_FITS = [get_fitness_with_additional_arguments(Minecraft2DFlatTownRandomLayoutFitness, index_to_use=i) for i in range(20)]
    a = 0
    b = 0
    D = {}
    for i, fit in enumerate(ALL_FITS):
        letter = _get_letter(a, b)
        print(letter)
        D[letter] = lambda LET: run_mcc(NUM, LET, level_class=Flat2DTownLevel, use_one_hot_encoding=False, additional_fitnesses=[fit], distance_function=visual_diversity_normalised, max_dist=1, pop_size=50, generations=(g:= 150), dimensions=3, NUM_SEEDS=NUM_SEEDS, do_padding_randomly=False ,number_of_random_variables=1, random_perturb_size=0, multiple_iterations=5, USE_RAY=False, all_weights=[0, 0, 1], generator_kwargs=dict(input_center_tile=True, start_level_with_default_tile=True,  progress_default_tile=0), num_levels=5, intra_novelty_neighbours=4, num_neighbours_novelty=1)
        
        if letter == letter_to_use: break
        b += 1
        if b >= 26:
            b = 0
            a += 1
        
    if letter_to_use == "None" or letter_to_use is None: return
    D[letter_to_use](letter_to_use)

if __name__ == '__main__':
    fire.Fire(main)