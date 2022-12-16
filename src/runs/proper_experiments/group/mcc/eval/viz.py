import glob
import fire
import numpy as np

from games.minecraft.new.minecraft_level import MinecraftLevel
from runs.proper_experiments.group.utils.game_utils import _glob_get_first, _glob_get_latest, get_generator_net_game_from_pickle

import minecraft_pb2_grpc
from minecraft_pb2 import *
import grpc
channel = grpc.insecure_channel('localhost:5001')
client = minecraft_pb2_grpc.MinecraftServiceStub(channel)

def main(experiment_name: str,
         max_seeds: int = 10,
         max_runs_per_seed: int = 5,
         level_size: int = 10):
    np.random.seed(42)
    directory = _glob_get_first(f'../results/experiments/pcgnn_{experiment_name}/*/PCGNN/')
    
    directory = _glob_get_latest(f'{directory}/*')
    total_seeds = len(glob.glob(f'{directory}/*/*/*/*.pbz2'))
    seeds_to_iterate = min(total_seeds, max_seeds)
    _size = 5
    try:
        GAP = level_size + 5
    except:
        GAP = 20
    
    SSSS = GAP * max(seeds_to_iterate, max_runs_per_seed) * 5
    client.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=-20, y=4, z=-20),
            max=Point(x=SSSS, y=255, z=SSSS)
        ),
        type=AIR
    ))
    print("Cleared, placing now")
    blocks_list = []
    for seed in range(seeds_to_iterate):
        p = _glob_get_first(f'{directory}/*/*/{seed}/*.pbz2')
        if level_size is None or level_size == 'None':
            shape = None
        else: 
            shape=(level_size, level_size, level_size)
        generator, net, game = get_generator_net_game_from_pickle(p, shape=shape)
        for index in range(max_runs_per_seed):
            level: MinecraftLevel = generator(net)
            X_START = GAP * seed
            Z_START = GAP * index
            Y_START = 0
            arr = level.map
            for i in range(arr.shape[0]):
                for k in range(arr.shape[2]):
                    for j in range(arr.shape[1]):
                        tile = level.minecraft_mapping[int(arr[i, arr.shape[1] - 1 -j, k])]
                        blocks_list.append(
                            Block(position=Point(x=X_START + i, y=4 + j + Y_START, z=Z_START + k), type=tile, orientation=NORTH),
                        )
        client.spawnBlocks(Blocks(blocks=blocks_list))
if __name__ == '__main__':
    fire.Fire(main)
    # main('3005-j', level_size=None)