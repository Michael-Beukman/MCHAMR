from dataclasses import dataclass
from tkinter import N
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
import neat
import numpy as np
from common.utils import mysavefig, remove_axis_ticks_keep_border
Vec = Tuple[int, int]

@dataclass(frozen=True)
class Tile:
    value: int
    
    def __mul__(self, other):
        return self
    
    def __rmul__(self, other):
        return self

    def __float__(self):
        return float(self.value)
    
    def __int__(self):
        return self.value
    
    def __lt__(self, other):
        return (self.value < other.value)
    
    def __gt__(self, other):
        return (self.value > other.value)
    
class Generator:
    tile_mapping: Dict[Tile, "Generator"]
    
    overall_size: Vec # the overall size of the creation, after everything is done
    tilemap_size: Vec # The size of this tilemap, without any composition
    sub_size: Vec = (1, 1, 1) # the size of the smaller subcomponents.
    
    def __init__(self,
                  tilemap_size: Vec, sub_size: Vec = (1, 1, 1), overall_size: Vec = None,
                  tile_mapping: Dict[str, "Generator"] = None, keep_height_1: bool = False) -> None:
        self.overall_size = overall_size
        self.tilemap_size = tilemap_size
        self.sub_size = sub_size
        self.tile_mapping = tile_mapping
        self.keep_height_1 = keep_height_1
        self._sort_out_size()
        
    
    def make_map(self) -> List[List[Tile]]:
        raise NotImplementedError()
    
    def set_size(self, tilemap_size: Vec = None, sub_size: Vec = None, overall_size: Vec = None):
        if tilemap_size is not None: self.tilemap_size = tilemap_size
        if sub_size is not None: self.sub_size = sub_size
        self.overall_size = overall_size
        
        self._sort_out_size()
    
    def _sort_out_size(self):
        if self.overall_size is None: 
            self.overall_size = (self.tilemap_size[0] * self.sub_size[0],
                                 self.tilemap_size[1] * self.sub_size[1],
                                 self.tilemap_size[2] * self.sub_size[2],
                                 )
        if self.tilemap_size is None or self.overall_size is not None:
            # print(self, self.overall_size, self.sub_size)
            assert self.overall_size[0] % self.sub_size[0] == 0, f"{self.overall_size[0]} vs {self.sub_size[0]}"
            assert self.overall_size[1] % self.sub_size[1] == 0, f"{self.overall_size[1]} vs {self.sub_size[1]}"
            assert self.overall_size[2] % self.sub_size[2] == 0, f"{self.overall_size[2]} vs {self.sub_size[2]}"
            self.tilemap_size = (self.overall_size[0] // self.sub_size[0],
                                 self.overall_size[1] // self.sub_size[1],
                                 self.overall_size[2] // self.sub_size[2],
                                 )
        if self.keep_height_1:
            self.overall_size = (self.overall_size[0], 1, self.overall_size[-1])
        assert self.overall_size == (self.tilemap_size[0] * self.sub_size[0], 1 if self.keep_height_1 else self.tilemap_size[1] * self.sub_size[1], self.tilemap_size[2] * self.sub_size[2])
        

class DefaultGenerator(Generator):
    def __init__(self, tile: Tile, overall_size: Vec, **kwargs) -> None:
        super().__init__(tilemap_size=overall_size, overall_size=overall_size, tile_mapping=None, **kwargs)
        self.tile = tile
    
    def make_map(self) -> List[List[Tile]]:
        return np.ones(self.overall_size) * self.tile

def get_all_rectangles(arr: np.ndarray, do_coalesce: bool = True):
    arr = arr.astype(np.int32)
    uniques = np.unique(arr)
    rects = {}
    for u in uniques:
        temp = arr == u
        rects_here = []
        added_coords_already = set()
        def is_rect(y, x, z, dy, dx, dz):
            if y + dy > temp.shape[0] or x + dx > temp.shape[1] or z + dz > temp.shape[2]: return False
            
            return np.sum(temp[y:y+dy, x:x+dx, z: z+dz]) == dy * dx * dz
        
        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                for k in range(temp.shape[2]):
                    if temp[i, j, k] == 0: continue
                    if(i, j, k) in added_coords_already: continue
                    # this is the start of a new rectangle:
                    shift_x = 1
                    shift_y = 1
                    shift_z = 1
                    can_check = do_coalesce
                    while can_check:
                        can_check = False
                        if is_rect(i, j, k, shift_y + 1, shift_x, shift_z):
                            shift_y += 1
                            can_check = True
                        if is_rect(i, j, k, shift_y, shift_x+1, shift_z):
                            shift_x += 1
                            can_check = True
                        if is_rect(i, j, k, shift_y, shift_x, shift_z + 1):
                            shift_z += 1
                            can_check = True
                        
                    # shift_x += 1; shift_y += 1
                    rects_here.append((i, j, k, shift_y, shift_x, shift_z))
                    for ii in range(shift_y):
                        for jj in range(shift_x):
                            for kk in range(shift_z):
                                added_coords_already.add((i + ii, j + jj, k + kk))
        rects[Tile(int(u))] = rects_here
    return rects


def generate(gen: Generator, size=None, do_coalesce: bool = True) -> List[List[List[Tile]]]:
    if size is not None: gen.set_size(overall_size=size)
    map = gen.make_map()
    new_filled_out_map = np.zeros((gen.overall_size[0], gen.overall_size[1], gen.overall_size[2]), dtype=Tile)
    if gen.tile_mapping is None: return map
    all_rectangles = get_all_rectangles(map, do_coalesce=do_coalesce)
    
    for tile_type, rects in all_rectangles.items():
        for (y, x, z, h, w, d) in rects:
            new_generator = gen.tile_mapping[tile_type]
            new_map = generate(new_generator, size := (h * gen.sub_size[0], w * gen.sub_size[1], d * gen.sub_size[2]), do_coalesce)
            
            if (ns := new_map.shape) != size:
                if ns[0] == size[0] and ns[-1] == size[-1] and (t := ns[1]) in [1, 2]:
                    tmp = np.zeros(size, dtype=new_map.dtype) 
                    tmp[:] = Tile(7) # If a level is too small, fill it with air
                    tmp[:, :t, :] = new_map
                    new_map = tmp
                if new_map.shape != size and tuple(reversed(new_map.shape)) == size:
                    new_map = np.transpose(new_map, axes=[2, 1, 0])
                assert new_map.shape == size, f"{new_map.shape} != {size}!"
            new_filled_out_map[y * gen.sub_size[0]: (y + h) * gen.sub_size[0], 
                               x * gen.sub_size[1]: (x + w) * gen.sub_size[1],
                               z * gen.sub_size[2]: (z + d) * gen.sub_size[2],
                               ] = new_map
    return new_filled_out_map