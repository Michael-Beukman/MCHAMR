from typing import Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
from games.level import Level
from games.minecraft.blocks import AIR, BLOCKS_TO_USE_REV, BRICK_BLOCK, COBBLESTONE, DIAMOND_BLOCK, EMERALD_BLOCK, GOLD_BLOCK, GRASS, IRON_BLOCK
from games.minecraft.new.minecraft_level import MinecraftLevel


class Flat2DTownLevel(MinecraftLevel):
    """
        A town level with a flat level structure.
    """
    def __init__(self, width=25, height=1, depth=25):
        super().__init__(width, height, tile_types={
            0: 'road',
            1: 'air',
            2: 'wall',
        })
        self.depth = depth
        self.map = np.zeros((width, height, depth))
        self.minecraft_mapping = [COBBLESTONE, AIR, DIAMOND_BLOCK]