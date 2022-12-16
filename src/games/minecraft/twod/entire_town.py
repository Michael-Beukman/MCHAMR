from typing import Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
from games.level import Level
from games.minecraft.blocks import AIR, BLOCKS_TO_USE_REV, BRICK_BLOCK, COBBLESTONE, DIAMOND_BLOCK, EMERALD_BLOCK, GOLD_BLOCK, GRASS, IRON_BLOCK, SAPLING, WATER, YELLOW_FLOWER
from games.minecraft.new.minecraft_level import MinecraftLevel


class Minecraft2DEntireTownLevel(MinecraftLevel):
    """
        This is the entire town level, consisting of all of the different subtiles
    """
    def __init__(self, width=50, height=1, depth=50):
        super().__init__(width, height, tile_types={
            0: 'road',

            1: 'wall',
            2: 'open',
            3: 'bed',
            
            4: 'grass',
            5: 'flower',
            6: 'tree',
            7: 'water',
        })
        self.depth = depth
        self.map = np.zeros((width, height, depth))
        self.minecraft_mapping = [COBBLESTONE, BRICK_BLOCK, AIR, GOLD_BLOCK, GRASS, YELLOW_FLOWER, SAPLING, WATER]