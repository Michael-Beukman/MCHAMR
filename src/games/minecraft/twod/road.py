from typing import Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
from games.level import Level
from games.minecraft.blocks import AIR, BLOCKS_TO_USE_REV, BRICK_BLOCK, COBBLESTONE, DIAMOND_BLOCK, EMERALD_BLOCK, GOLD_BLOCK, GRASS, IRON_BLOCK
from games.minecraft.new.minecraft_level import MinecraftLevel
from minecraft_pb2 import WATER


class Minecraft2DRoadLevel(MinecraftLevel):
    """
        Basically a highway
    """
    def __init__(self, width=10, height=1, depth=10):
        super().__init__(width, height, tile_types={
            0: 'garden',
            1: 'road',
            2: 'water',
        })
        self.depth = depth
        self.map = np.zeros((width, height, depth))
        self.minecraft_mapping = [GRASS, COBBLESTONE, WATER]