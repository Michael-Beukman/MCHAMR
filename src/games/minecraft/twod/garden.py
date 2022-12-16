from typing import Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
from games.level import Level
from games.minecraft.blocks import AIR, BLOCKS_TO_USE_REV, BRICK_BLOCK, COBBLESTONE, DIAMOND_BLOCK, EMERALD_BLOCK, GOLD_BLOCK, GRASS, IRON_BLOCK, SAPLING, WATER, YELLOW_FLOWER
from games.minecraft.new.minecraft_level import MinecraftLevel


class Minecraft2DGardenLevel(MinecraftLevel):
    """
        A simple garden level
    """
    def __init__(self, width=5, height=1, depth=5):
        super().__init__(width, height, tile_types={
            0: 'grass',
            1: 'flower',
            2: 'tree',
            3: 'water',
        })
        self.depth = depth
        self.map = np.zeros((width, height, depth))
        self.minecraft_mapping = [GRASS, YELLOW_FLOWER, SAPLING, WATER]