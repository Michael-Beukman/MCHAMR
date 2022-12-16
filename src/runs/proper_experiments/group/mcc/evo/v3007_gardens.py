import fire
from contrib.fitness.minecraft.garden.coherent_garden_fitness import CoherentGardenFitness
from games.minecraft.new.garden.minecraft_garden_level import MinecraftGardenLevel

from novelty_neat.novelty.distance_functions.distance import minecraft_js_divergence_tile_distros_matthew,visual_diversity_normalised
from runs.proper_experiments.group.mcc.michael.mcc_common import run_mcc


def main(letter):
    """
        This tries to reproduce the evocraft gardens
    """
    
    {
        'a': lambda: run_mcc(3007, letter, level_class=MinecraftGardenLevel, use_one_hot_encoding=False, additional_fitnesses=[CoherentGardenFitness], distance_function=visual_diversity_normalised, max_dist=1, pop_size=20, generations=100, all_weights=[1, 1, 4], do_padding_randomly=False, number_of_random_variables=2, random_perturb_size=0.1565),
    }[letter]()

if __name__ == '__main__':
    fire.Fire(main)