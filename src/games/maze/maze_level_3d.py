from typing import Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
from games.level import Level
from games.minecraft.new.minecraft_level import MinecraftLevel
from minecraft_pb2 import AIR, OBSIDIAN


class MazeLevel3D(MinecraftLevel):
    """A simple level consisting of only empty and filled blocks
    """
    def __init__(self, width=10, height=1, depth=10,
                        start: Union[Tuple[int, int], None] = None,
                        end: Union[Tuple[int, int], None]  = None):

        super().__init__(width, height, tile_types={
            0: 'empty',
            1: 'filled',
        })
        self.depth = depth
        self.map = np.zeros((width, height, depth))
        self.start = start if start is not None else (0, 0)
        self.end = end if end is not None else  (width - 1, height - 1)
        self.minecraft_mapping = [AIR, OBSIDIAN]
