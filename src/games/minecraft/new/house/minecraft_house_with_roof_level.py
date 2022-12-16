from typing import Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
from games.level import Level
from games.minecraft.blocks import AIR, BLOCKS_TO_USE_REV, BRICK_BLOCK, PLANKS
from games.minecraft.new.minecraft_level import MinecraftLevel


class MinecraftHouseWithRoofLevel(MinecraftLevel):
    """A simple house consisting of empty and filled tiles
    """
    def __init__(self, width=10, height=10, depth=10):
        super().__init__(width, height, tile_types={
            0: 'empty',
            1: 'brick',
            2: 'roof',
        })
        self.depth = depth
        self.map = np.zeros((width, height, depth))
        self.minecraft_mapping = [AIR, BRICK_BLOCK, PLANKS]