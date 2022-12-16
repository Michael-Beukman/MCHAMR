import numpy as np

from common.utils import remove_axis_ticks_keep_border
from games.level import Level
from games.maze.maze_level_3d import MazeLevel3D
from games.minecraft.new.garden.minecraft_garden_level import MinecraftGardenLevel
from games.minecraft.twod.compose.composed_house import Composed2DHouseLevel
from games.minecraft.twod.compose.composed_town import Composed2DTownLevel
from games.minecraft.twod.compose.flat_town import Flat2DTownLevel
from games.minecraft.twod.entire_town import Minecraft2DEntireTownLevel
from games.minecraft.twod.garden import Minecraft2DGardenLevel
from games.minecraft.twod.house import Minecraft2DHouseLevel
from games.minecraft.twod.road import Minecraft2DRoadLevel
from games.minecraft.twod.town import Minecraft2DTownLevel, Minecraft2DTownLevel12Size
from novelty_neat.maze.utils import get_proper_maze_path
import seaborn as sns
from games.maze.maze_level import MazeLevel

PLOT_DIR = '../results/plots'


def plot_single_level(level: Level, ax, annot=True, rasterized=False):
    if isinstance(level, MazeLevel) or isinstance(level, MazeLevel3D):
        if isinstance(level, MazeLevel3D): level.map = level.map[:, 0]
        ax.imshow(1 - level.map, vmin=0, vmax=1, cmap='gray')
        path = get_proper_maze_path(level)
        if path is not None:
            x, y = zip(*path)
            ax.plot(x, y)
            ax.set_title(f"{len(path)}")
        return
    remove_axis_ticks_keep_border(ax)
    assert level.map.shape[1] == 1
    level.map = level.map[:, 0, :]
    img = level.map
    
    labels = []
    for row in img:
        temp = []
        for col in row:
            s = level.tile_types[int(col)]
            if '_' in s:
                s = s[0] + s[-1]
            else:
                s = s[0]
            s = s.upper()
            temp.append(s)
        labels.append(temp)
        
    if isinstance(level, MinecraftGardenLevel):
        cmap = ['green', 'brown', 'red', 'yellow', 'blue']
    elif isinstance(level, Minecraft2DTownLevel):
        cmap = ['green', 'grey', 'blue']
    elif isinstance(level, Minecraft2DTownLevel12Size):
        cmap = ['green', 'grey', 'blue']
    elif isinstance(level, Minecraft2DHouseLevel):
        cmap = ['red', 'white', 'purple']
    elif isinstance(level, Minecraft2DRoadLevel):
        cmap = ['green', 'grey', 'blue']
    elif isinstance(level, Minecraft2DGardenLevel):
        cmap = ['green', 'yellow', 'brown', 'blue']
    
    elif isinstance(level, Minecraft2DEntireTownLevel):
        cmap = ['grey', 'red', 'white', 'purple', 'green', 'yellow', 'beige', 'blue']
    elif isinstance(level, MazeLevel3D):
        cmap = ['white', 'black']

    elif isinstance(level, Composed2DTownLevel):
        cmap = ['grey', 'blue']
    elif isinstance(level, Composed2DHouseLevel):
        cmap = ['white', 'brown']
    elif isinstance(level, Flat2DTownLevel):
        cmap = ['grey', 'white', 'brown']
    else:
        assert 0, level;
    new_cmap = []
    for tile in np.unique(img):
        tile = int(tile)
        new_cmap.append(cmap[tile])
    if annot==True:
        annot = labels
    ax = sns.heatmap(img, cmap=cmap, annot=annot, annot_kws={'fontsize': 12}, fmt='s', ax=ax, vmin=0, vmax=len(cmap), cbar=False, rasterized=rasterized)

    remove_axis_ticks_keep_border(ax)