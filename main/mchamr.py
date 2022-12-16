
import copy
from dataclasses import dataclass
from tkinter import N
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
import neat
from common.utils import mysavefig, remove_axis_ticks_keep_border
from contrib.theory.place_in_mc import place_in_minecraft
from contrib.theory.base_mchamr import Generator, Tile, Vec, generate, DefaultGenerator
from contrib.theory.mchamr import PCGNNGenerator, GardenGenerator
from games.level import Level
from novelty_neat.generation import NeatLevelGenerator
from runs.proper_experiments.group.utils.game_utils import get_generator_net_game_from_pickle, get_generator_net_game_from_pickle_map_elites, get_level_with_same_class_different_shape, get_pickle_name_from_exp_name
import numpy as np
import seaborn as sns
from games.minecraft.blocks import AIR, BRICK_BLOCK, COBBLESTONE, DIAMOND_BLOCK, EMERALD_BLOCK, GOLD_BLOCK, GRASS, LOG, LOG2, OBSIDIAN, RED_FLOWER, SANDSTONE, SAPLING, STAINED_HARDENED_CLAY, STONEBRICK, WATER, WOODEN_SLAB, YELLOW_FLOWER, PLANKS
import fire
sns.set_theme()
# Some setup
TILES_TO_NAMES: Dict["Tile", str] = {
    Tile(0): 'empty',
    Tile(1): 'garden',
    Tile(2): 'road',
    Tile(3): 'house_1',
    Tile(4): 'house_2',
    Tile(5): 'house_3',
    Tile(6): 'brick',
    
    
    Tile(7): 'air',
    Tile(8): 'oak',
    Tile(9): 'red flower',
    Tile(10): 'yellow flower',
    Tile(11): 'water',
    Tile(12): 'house',
    Tile(13): 'grass',
    Tile(14): 'roof',
}
COLORS = ['black', 'green', 'grey', 'orange', 'purple', 'pink', 'brown', 'green', 'teal', 'red', 'yellow', 'blue', 'orange']
NAMES_TO_GENERATORS: Dict[str, "Generator"] = {}

NAMES_TO_MINECRAFT_BLOCKS = {
    'empty': AIR,
    'garden': GRASS,
    'road': STONEBRICK,
    'house_1': DIAMOND_BLOCK,
    'house_2': EMERALD_BLOCK,
    'house_3': GOLD_BLOCK,
    'brick': BRICK_BLOCK,
    'air': AIR,
    'oak': SAPLING,
    'red flower': RED_FLOWER,
    'yellow flower': YELLOW_FLOWER,
    'water': WATER,
    'house':  DIAMOND_BLOCK,
    
    'grass':  GRASS,
    'roof':  STAINED_HARDENED_CLAY,
}

TILES_TO_MINECRAFT_BLOCKS = {tile: NAMES_TO_MINECRAFT_BLOCKS[name] for tile, name in TILES_TO_NAMES.items()}

def clean_name(s):
    if 'flower' in s:
        return 'F'
    if 'house_' in s: return f'H{s[-1]}'
    if s == 'empty': return 'E'
    return s

def convert_array_to_minecraft_blocks(arr: np.ndarray) -> np.ndarray:
    # This converts the array so that each integer represents a minecraft block.
    new_arr = np.zeros_like(arr, dtype=np.int32)
    for i in np.unique(arr):
        new_arr[arr == i] = TILES_TO_MINECRAFT_BLOCKS[i]
    return new_arr

def plot_level(img, save_name, simple_first, TILES_TO_NAMES, COLORS, annot=True, maze=False):
    img = img[:, 0]
    plt.figure(figsize=(12.5, 12.5))
    labels = []
    for row in img:
        temp = []
        for col in row:
            temp.append(clean_name(TILES_TO_NAMES[col]))
        labels.append(temp)
    img = img.astype(np.int32) # due to weird seaborn things
    if maze:
        plt.imshow(1 - img, vmin=0, vmax=1, cmap='gray')
        ax = plt.gca()
    else:
        annot = labels if annot else annot
        ax = sns.heatmap(img, cmap=COLORS, annot=annot, annot_kws={'fontsize': 12}, fmt='s', vmin=0, vmax=len(COLORS))
    remove_axis_ticks_keep_border(ax)
    fname = f'plots/{save_name}_{simple_first}.jpg'
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    mysavefig(fname, do_tight_layout=False)
    print('saved to', fname)


def do_mcc(gen: Generator, save_name, only_plot=False, do_coalesce=True):
    img = generate(gen, do_coalesce=do_coalesce)
    if only_plot:
        plot_level(img, save_name, only_plot, TILES_TO_NAMES, COLORS)
    else:
        minecraft_arr = convert_array_to_minecraft_blocks(img)
        place_in_minecraft(minecraft_arr)


def get_generator(exp_name, tilemap_size, seed=0, should_be_garden=False, **kwargs):
    generator, net, game, dic = get_generator_net_game_from_pickle(get_pickle_name_from_exp_name(exp_name, seed=seed), shape=tilemap_size ,return_entire_dic=True)
    if should_be_garden:
        return GardenGenerator(generator, net, game.level, tilemap_size, keep_height_1=True, **kwargs)
    return PCGNNGenerator(generator, net, game.level, tilemap_size, **kwargs)
    
def main():
    T = 5

    city_sub_size = (25, 5, 25)
    town_sub_size = (T, 5, T)
    city_tilemap_size = (10, 1, 10)
    
    GARDEN_EXP = '3007-a'
    HOUSE_EXP = '3602-aab'
    TOWN_EXP = '3551-cfa'
    CITY_EXP = '3551-dfa'
    
    
    # This defines which values get plotted.
    np.random.seed(42)
    NAMES_TO_GENERATORS['house']  = get_generator(HOUSE_EXP, tilemap_size=town_sub_size)
    NAMES_TO_GENERATORS['garden'] = get_generator(GARDEN_EXP, tilemap_size=(T, 1, T), should_be_garden=True)
    town_gen = get_generator(TOWN_EXP, seed=9, tilemap_size=tuple(c // t for c, t in zip(city_sub_size,town_sub_size)), sub_size=town_sub_size, NAMES_TO_GENERATORS=NAMES_TO_GENERATORS)
    new_names = copy.deepcopy(NAMES_TO_GENERATORS)
    new_names['house'] = town_gen
    
    city_gen = get_generator(CITY_EXP, seed=4, tilemap_size=city_tilemap_size, sub_size=city_sub_size, NAMES_TO_GENERATORS=NAMES_TO_GENERATORS | {'house': town_gen})
    do_mcc(city_gen, 'mcc2_3d', only_plot=False, do_coalesce=True)


if __name__ == '__main__':
    main()