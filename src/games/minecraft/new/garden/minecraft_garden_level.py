import numpy as np
from games.minecraft.blocks import DIAMOND_BLOCK, EMERALD_BLOCK, LAPIS_BLOCK, PLANKS, REDSTONE_BLOCK
from games.minecraft.new.minecraft_level import MinecraftLevel


class MinecraftGardenLevel(MinecraftLevel):
    def __init__(self, width=10, height=1, depth=10):
       
        super().__init__(width, height, tile_types={
            0: 'air', # change dirt block to grass (pull one block more done)
            1: 'oak',
            2: 'red flower',
            3: 'yellow flower',
            4: 'water', # place block below it to be water (pull one block more down)
        })
        self.depth = depth
        self.map = np.zeros((width, height, depth))
        self.minecraft_mapping = [DIAMOND_BLOCK, PLANKS, REDSTONE_BLOCK, EMERALD_BLOCK, LAPIS_BLOCK]