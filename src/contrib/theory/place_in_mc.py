
import time
from minecraft_pb2 import *
import numpy as np
def place_in_minecraft(world: np.ndarray):
    import grpc
    import tqdm

    import minecraft_pb2_grpc

    channel = grpc.insecure_channel('localhost:5001')
    client = minecraft_pb2_grpc.MinecraftServiceStub(channel)


    client.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=-10, y=0, z=-10),
            max=Point(x=10, y=3, z=10)
        ),
        type=GRASS
    ))
    def clear_all(min=-1000, max = 1000):
        X = min
        L = 100
        while X < max:
            Z = min
            while Z < max:
                client.fillCube(FillCubeRequest(
                    cube=Cube(
                        min=Point(x=X, y=2, z=Z),
                        max=Point(x=X + L, y=3, z=Z + L)
                    ),
                    type=GRASS
                ))
                client.fillCube(FillCubeRequest(
                    cube=Cube(
                        min=Point(x=X, y=4, z=Z),
                        max=Point(x=X + L, y=30, z=Z + L)
                    ),
                    type=AIR
                ))
                Z += L
                time.sleep(0.1)
            X += L
    print("Clearing - might take a while")            
    clear_all(-500, 500);
    print("Cleared an area")            
    print(f"Generating a world of shape {world.shape} now")
    S = 15
    X_START = 0
    blocks_list = []
    COORDS = np.argwhere(world != AIR)
    for i, j, k in tqdm.tqdm(COORDS):
        blocks_list.append(
        Block(position=Point(x=X_START + i - 5, y=4 + j, z=k - 5), type=int(world[i, j, k]), orientation=NORTH),
        )
        if len(blocks_list) >= 1000:
            client.spawnBlocks(Blocks(blocks=blocks_list))
            blocks_list = []
            time.sleep(2)
    client.spawnBlocks(Blocks(blocks=blocks_list))