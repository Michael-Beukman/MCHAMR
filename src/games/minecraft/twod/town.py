from typing import Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
from games.level import Level
from games.minecraft.blocks import AIR, BLOCKS_TO_USE_REV, BRICK_BLOCK, COBBLESTONE, DIAMOND_BLOCK, EMERALD_BLOCK, GOLD_BLOCK, GRASS, IRON_BLOCK
from games.minecraft.new.minecraft_level import MinecraftLevel


class Minecraft2DTownLevel(MinecraftLevel):
    """
        A simple town level
    """
    def __init__(self, width=10, height=1, depth=10):
        super().__init__(width, height, tile_types={
            0: 'garden',
            1: 'road',
            2: 'house',
        })
        self.depth = depth
        self.map = np.zeros((width, height, depth))
        self.minecraft_mapping = [GRASS, COBBLESTONE, DIAMOND_BLOCK]
        
class Minecraft2DTownLevel12Size(MinecraftLevel):
    """
        A simple town level
    """
    def __init__(self, width=12, height=1, depth=12):
        super().__init__(width, height, tile_types={
            0: 'garden',
            1: 'road',
            2: 'house',
        })
        self.depth = depth
        self.map = np.zeros((width, height, depth))
        self.minecraft_mapping = [GRASS, COBBLESTONE, DIAMOND_BLOCK]
        
class Minecraft2DTownLevel20Size(Minecraft2DTownLevel):
    def __init__(self, width=20, height=1, depth=20):
        super().__init__(width, height, depth)
        
class Minecraft2DTownLevel30Size(Minecraft2DTownLevel):
    def __init__(self, width=30, height=1, depth=30):
        super().__init__(width, height, depth)
        
        
class Minecraft2DTownLevel40Size(Minecraft2DTownLevel):
    def __init__(self, width=40, height=1, depth=40):
        super().__init__(width, height, depth)

class Minecraft2DTownLevel50Size(Minecraft2DTownLevel):
    def __init__(self, width=50, height=1, depth=50):
        super().__init__(width, height, depth)

class Minecraft2DTownLevel100Size(Minecraft2DTownLevel):
    def __init__(self, width=100, height=1, depth=100):
        super().__init__(width, height, depth)