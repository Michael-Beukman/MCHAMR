import copy
from dataclasses import dataclass
from tkinter import N
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
import neat
from common.utils import mysavefig, remove_axis_ticks_keep_border
from contrib.theory.place_in_mc import place_in_minecraft
from contrib.theory.base_mchamr import Generator, Tile, Vec, generate, DefaultGenerator
from games.level import Level
from novelty_neat.generation import NeatLevelGenerator
from runs.proper_experiments.group.utils.game_utils import get_generator_net_game_from_pickle, get_generator_net_game_from_pickle_map_elites, get_level_with_same_class_different_shape, get_pickle_name_from_exp_name
import numpy as np
import seaborn as sns
from games.minecraft.blocks import AIR, BRICK_BLOCK, COBBLESTONE, DIAMOND_BLOCK, EMERALD_BLOCK, GOLD_BLOCK, GRASS, LOG, LOG2, OBSIDIAN, RED_FLOWER, SANDSTONE, SAPLING, STAINED_HARDENED_CLAY, STONEBRICK, WATER, WOODEN_SLAB, YELLOW_FLOWER, PLANKS
sns.set_theme()
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
NAMES_TO_TILES = {v: k for k, v in TILES_TO_NAMES.items()}
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
    'roof':  STAINED_HARDENED_CLAY, # STAINED_HARDENED_CLAY not bad
}

TILES_TO_MINECRAFT_BLOCKS = {tile: NAMES_TO_MINECRAFT_BLOCKS[name] for tile, name in TILES_TO_NAMES.items()}

class PCGNNGenerator(Generator):
    def __init__(self,
                 pcgnn_generator: NeatLevelGenerator, net: neat.nn.FeedForwardNetwork, level: Level,
                 tilemap_size: Vec, sub_size: Vec = (1, 1, 1), overall_size: Vec = None, tile_mapping: Dict[Tile, "Generator"] = None, 
                 NAMES_TO_TILES=NAMES_TO_TILES, NAMES_TO_GENERATORS=NAMES_TO_GENERATORS, TILES_TO_NAMES=TILES_TO_NAMES,
                 is_2d_generator: bool = False,
                 **kwargs) -> None:
        assert tile_mapping is None
        self.NAMES_TO_TILES = NAMES_TO_TILES
        self.NAMES_TO_GENERATORS = NAMES_TO_GENERATORS
        self.is_2d_generator = is_2d_generator
        if tile_mapping is None:
            countin = 0
            countout = 0
            new_mapping = {}
            for name in level.tile_types_reversed:
                if name in NAMES_TO_GENERATORS:
                    countin += 1
                    new_mapping[NAMES_TO_TILES[name]] = NAMES_TO_GENERATORS[name]
                else:
                    countout += 1
            if countin != 0 and countout != 0:
                for name in level.tile_types_reversed:
                    tile = NAMES_TO_TILES[name]
                    subsize_one_d = (self.sub_size[0], 1, self.sub_size[-2])
                    if tile not in new_mapping:
                        new_mapping[tile] = DefaultGenerator(NAMES_TO_TILES[name], subsize_one_d, keep_height_1=True)
                        
            if countin != 0: 
                tile_mapping = new_mapping

        super().__init__(tilemap_size, sub_size, overall_size, tile_mapping, **kwargs)
        self.pcgnn_generator = pcgnn_generator
        self.net = net
        self.level = level

        if self.tile_mapping is not None:
            good_str = [TILES_TO_NAMES[t] for t in self.tile_mapping.keys()]
            assert sorted(good_str) == sorted(self.level.tile_types_reversed.keys())

    def set_size(self, tilemap_size: Vec = None, sub_size: Vec = None, overall_size: Vec = None):
        ans = super().set_size(tilemap_size, sub_size, overall_size)
        new_shape = self.tilemap_size
        if self.is_2d_generator and len(new_shape) == 3: new_shape = (new_shape[0], new_shape[-1])
        self.pcgnn_generator.game.level = get_level_with_same_class_different_shape(self.pcgnn_generator.game, new_shape, should_keep_middle_dim_size_if_1=True)
        self.level = self.pcgnn_generator.game.level
        
        return ans
    
    def make_map(self) -> List[List[Tile]]:
        map = self.pcgnn_generator(self.net)

        # 3D generation
        map = map.map
        if len(map.shape) == 2:
            map = map[:, None]
        map = map[:, ::-1]
        map_of_tiles = []
        for row in map:
            temp = []
            for col in row:
                temp2 = []
                for zcol in col:
                    zcol = int(zcol)
                    name_of_current_tile: str = self.level.tile_types[zcol]
                    tile_value_current: Tile = self.NAMES_TO_TILES[name_of_current_tile]
                    assert type(tile_value_current) != int
                    temp2.append(tile_value_current)
                temp.append(temp2)
            map_of_tiles.append(temp)
        return np.array(map_of_tiles)
class GardenGenerator(PCGNNGenerator):
    def __init__(self, pcgnn_generator: NeatLevelGenerator, net: neat.nn.FeedForwardNetwork, level: Level, tilemap_size: Vec, sub_size: Vec = (1, 1, 1), overall_size: Vec = None, tile_mapping: Dict[Tile, "Generator"] = None, **kwargs) -> None:
        super().__init__(pcgnn_generator, net, level, tilemap_size, sub_size, overall_size, tile_mapping, **kwargs)
    
    def make_map(self) -> List[List[Tile]]:
        ans = super().make_map()
        new = np.zeros((ans.shape[0], ans.shape[1] +1, ans.shape[2]), dtype=ans.dtype)
        idx_water = ans == NAMES_TO_TILES['water']
        new[:, 0, :] = NAMES_TO_TILES['grass']
        new[:, :1][idx_water] = NAMES_TO_TILES['water']
        ans[idx_water] = NAMES_TO_TILES['air']
        
        ans[ans == NAMES_TO_TILES['grass']] = NAMES_TO_TILES['air']
        
        new[:, 1:, :] = ans
        return new
        

def clean_name(s):
    if 'flower' in s:
        return 'F'
    if 'house_' in s: return f'H{s[-1]}'
    if s == 'empty': return 'E'
    return s

def convert_array_to_minecraft_blocks(arr: np.ndarray) -> np.ndarray:
    # This converts the array so that each integer represents a minecraft block.
    new_arr = np.zeros_like(arr, dtype=np.int32)
    print(arr)
    for i in np.unique(arr):
        idx = arr == i
        new_val = TILES_TO_MINECRAFT_BLOCKS[i]
        new_arr[idx] = new_val        
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
    

    print(len(COLORS), img.max())
    if maze:
        plt.imshow(1 - img, vmin=0, vmax=1, cmap='gray')
        ax = plt.gca()
    else:
        annot = labels if annot else annot
        ax = sns.heatmap(img, cmap=COLORS, annot=annot, annot_kws={'fontsize': 12}, fmt='s', vmin=0, vmax=len(COLORS))
    remove_axis_ticks_keep_border(ax)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()


def do_mcc(gen: Generator, save_name, simple_first=False, do_coalesce=True):
    if simple_first:
        img = (gen).make_map()
    else:
        img = generate(gen, do_coalesce=do_coalesce)
    if img.shape != (10, 1, 10) and img.shape[1] != 1:
        minecraft_arr = convert_array_to_minecraft_blocks(img)
        place_in_minecraft(minecraft_arr)
    else:
        plot_level(img, save_name, simple_first, TILES_TO_NAMES, COLORS)



def v20(simple_first):
    np.random.seed(42)
    
    generator, net, game, dic = get_generator_net_game_from_pickle(get_pickle_name_from_exp_name('3602-aab'), shape=(4, 4, 4) ,return_entire_dic=True)
    house_gen = PCGNNGenerator(generator, net, game.level, (2, 2, 2))

    generator, net, game, dic = get_generator_net_game_from_pickle(get_pickle_name_from_exp_name('3007-a'), shape=(4, 1 ,4) ,return_entire_dic=True)
    garden_gen = GardenGenerator(generator, net, game.level, (4, 1, 4), keep_height_1=True)

    NAMES_TO_GENERATORS['house'] = house_gen
    NAMES_TO_GENERATORS['house_1'] = house_gen
    NAMES_TO_GENERATORS['house_2'] = house_gen
    NAMES_TO_GENERATORS['house_3'] = house_gen
    NAMES_TO_GENERATORS['garden'] = garden_gen
    NAMES_TO_GENERATORS['empty']  = house_gen
    TOWN_SIZE = (5, 1, 5)
    generator, net, game, dic = get_generator_net_game_from_pickle(get_pickle_name_from_exp_name('3551-cfa', seed=9), shape=TOWN_SIZE, return_entire_dic=True)
    np.random.seed(42)
    
    city_sub_size = (16, 5, 16)
    town_sub_size = (4, 5, 4)
    
    town_gen = PCGNNGenerator(generator, net, game.level, tilemap_size=tuple(c // t for c, t in zip(city_sub_size,town_sub_size)), sub_size=town_sub_size)
    
    
    new_names = copy.deepcopy(NAMES_TO_GENERATORS)
    new_names['house'] = town_gen
    city_tilemap_size = (12, 1, 12)
    generator, net, game, dic = get_generator_net_game_from_pickle(get_pickle_name_from_exp_name('3551-dfa', seed=4), shape=city_tilemap_size, return_entire_dic=True)
    
    new_names['house']      = town_gen
    new_names['garden']     = garden_gen
    new_names['empty']      = town_gen
    
    city_gen = PCGNNGenerator(generator, net, game.level, tilemap_size=city_tilemap_size, sub_size=city_sub_size, NAMES_TO_GENERATORS=new_names)

    do_mcc(city_gen, 'mcc2_3d', simple_first, do_coalesce=True)

def v23(simple_first):
    np.random.seed(42)
    
    generator, net, game, dic = get_generator_net_game_from_pickle(get_pickle_name_from_exp_name('3602-aab'), shape=(4, 4, 4) ,return_entire_dic=True)
    house_gen = PCGNNGenerator(generator, net, game.level, (2, 2, 2))

    generator, net, game, dic = get_generator_net_game_from_pickle(get_pickle_name_from_exp_name('3007-a'), shape=(4, 1 ,4) ,return_entire_dic=True)
    garden_gen = GardenGenerator(generator, net, game.level, (4, 1, 4), keep_height_1=True)

    NAMES_TO_GENERATORS['house'] = house_gen
    NAMES_TO_GENERATORS['house_1'] = house_gen
    NAMES_TO_GENERATORS['house_2'] = house_gen
    NAMES_TO_GENERATORS['house_3'] = house_gen
    NAMES_TO_GENERATORS['garden'] = garden_gen
    NAMES_TO_GENERATORS['empty']  = house_gen
    TOWN_SIZE = (5, 1, 5)
    generator, net, game, dic = get_generator_net_game_from_pickle(get_pickle_name_from_exp_name('3551-cfa', seed=9), shape=TOWN_SIZE, return_entire_dic=True)
    np.random.seed(42)
    
    city_sub_size = (16, 5, 16)
    town_sub_size = (4, 5, 4)
    
    town_gen = PCGNNGenerator(generator, net, game.level, tilemap_size=tuple(c // t for c, t in zip(city_sub_size,town_sub_size)), sub_size=town_sub_size)
    
    
    new_names = copy.deepcopy(NAMES_TO_GENERATORS)
    new_names['house'] = town_gen

    city_tilemap_size = (12, 1, 12)
    generator, net, game, dic = get_generator_net_game_from_pickle(get_pickle_name_from_exp_name('3551-dfa', seed=4), shape=city_tilemap_size, return_entire_dic=True)

    new_names['house']      = town_gen
    new_names['garden']     = garden_gen
    new_names['empty']      = town_gen
    
    city_gen = PCGNNGenerator(generator, net, game.level, tilemap_size=city_tilemap_size, sub_size=city_sub_size, NAMES_TO_GENERATORS=new_names)

    do_mcc(city_gen, 'mcc2_3d', simple_first, do_coalesce=False)

def v24(simple_first):
    np.random.seed(42)
    
    generator, net, game, dic = get_generator_net_game_from_pickle(get_pickle_name_from_exp_name('3602-aab'), shape=(4, 4, 4) ,return_entire_dic=True)
    house_gen = PCGNNGenerator(generator, net, game.level, (2, 2, 2))

    generator, net, game, dic = get_generator_net_game_from_pickle(get_pickle_name_from_exp_name('3007-a'), shape=(4, 1 ,4) ,return_entire_dic=True)
    garden_gen = GardenGenerator(generator, net, game.level, (4, 1, 4), keep_height_1=True)

    NAMES_TO_GENERATORS['house'] = house_gen
    NAMES_TO_GENERATORS['garden'] = garden_gen
    TOWN_SIZE = (5, 1, 5)
    generator, net, game, dic = get_generator_net_game_from_pickle(get_pickle_name_from_exp_name('3610-bbb'), shape=TOWN_SIZE, return_entire_dic=True)
    np.random.seed(42)
    town_gen = PCGNNGenerator(generator, net, game.level, tilemap_size=(6, 1, 6), sub_size=(5, 5, 5))
    
    new_names = copy.deepcopy(NAMES_TO_GENERATORS)
    new_names['house'] = town_gen
    
    generator, net, game, dic = get_generator_net_game_from_pickle(get_pickle_name_from_exp_name('3610-bbb', seed=0), shape=(8, 1, 8), return_entire_dic=True)
    
    city_gen = PCGNNGenerator(generator, net, game.level, tilemap_size=(8, 1, 8), sub_size=(30, 5, 30), NAMES_TO_GENERATORS=new_names)
    # print('town', town_gen.tile_mapping)
    do_mcc(city_gen, 'mcc2_3d', simple_first, do_coalesce=True)

def v25(simple_first):
    np.random.seed(42)
    
    generator, net, game, dic = get_generator_net_game_from_pickle(get_pickle_name_from_exp_name('3602-aab'), shape=(4, 4, 4) ,return_entire_dic=True)
    house_gen = PCGNNGenerator(generator, net, game.level, (2, 2, 2))

    generator, net, game, dic = get_generator_net_game_from_pickle(get_pickle_name_from_exp_name('3007-a'), shape=(4, 1 ,4) ,return_entire_dic=True)
    garden_gen = GardenGenerator(generator, net, game.level, (4, 1, 4), keep_height_1=True)

    NAMES_TO_GENERATORS['house'] = house_gen
    NAMES_TO_GENERATORS['garden'] = garden_gen
    NAMES_TO_GENERATORS['empty']  = house_gen
    TOWN_SIZE = (5, 1, 5)
    generator, net, game, dic = get_generator_net_game_from_pickle(get_pickle_name_from_exp_name('3551-cfa', seed=9), shape=TOWN_SIZE, return_entire_dic=True)
    np.random.seed(42)
    
    city_sub_size = (25, 5, 25)
    town_sub_size = (5, 5, 5)
    
    town_gen = PCGNNGenerator(generator, net, game.level, tilemap_size=tuple(c // t for c, t in zip(city_sub_size,town_sub_size)), sub_size=town_sub_size)
    
    
    new_names = copy.deepcopy(NAMES_TO_GENERATORS)
    new_names['house'] = town_gen
    city_tilemap_size = (10, 1, 10)
    generator, net, game, dic = get_generator_net_game_from_pickle(get_pickle_name_from_exp_name('3551-dfa', seed=4), shape=city_tilemap_size, return_entire_dic=True)

    new_names['house']      = town_gen
    new_names['garden']     = garden_gen
    
    city_gen = PCGNNGenerator(generator, net, game.level, tilemap_size=city_tilemap_size, sub_size=city_sub_size, NAMES_TO_GENERATORS=new_names)

    do_mcc(city_gen, 'mcc2_3d', simple_first, do_coalesce=True)


def main(simple_first=False):
    v24(simple_first)

if __name__ == '__main__':
    main(False)