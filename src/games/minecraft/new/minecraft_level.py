from typing import Dict, List, Union
from common.types import TileMap
from games.level import Level


class MinecraftLevel(Level):
    def __init__(self, width: int, height: int, tile_types: Dict[int, str] = ..., map: Union[None, TileMap] = None):
        super().__init__(width, height, tile_types, map)
        self.minecraft_mapping: List[int] = None
        
    