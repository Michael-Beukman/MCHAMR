import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from common.utils import mysavefig, remove_axis_ticks_keep_border
import seaborn as sns
sns.set_theme()
def main():

    a = np.random.randint(2, size=(10,10))
    a = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ])
    im = plt.imshow(a, cmap='Blues', interpolation='none', vmin=0, vmax=1, aspect='equal')

    def rect(pos):
        r = plt.Rectangle(pos-0.5, 1,1, facecolor="none", edgecolor="k", linewidth=2)
        plt.gca().add_patch(r)

    x,y = np.meshgrid(np.arange(a.shape[1]),np.arange(a.shape[0]))
    m = np.c_[x[a.astype(bool)],y[a.astype(bool)]]
    for pos in m:
        rect(pos)

    plt.show()
    
def main(yes):
    EDGE_COL     = "#38023B"
    COALESCE_COL = "#BBD5ED"
    COALESCE_COL = "#8CDEDC"
    ROAD_COL     = "#E6E6EA"
    ROAD_COL     = "#545863"
    ROAD_COL     = "#6D6A75"
    ROAD_COL     = "#BFBDC1"
    a = np.random.randint(2, size=(10,10))
    a = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])
    cmap = colors.ListedColormap(list_of_colours := [ROAD_COL, '#A288E3'])
    bounds=[0,1,2]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # im = plt.imshow(a, cmap='Blues', interpolation='none', vmin=0, vmax=1, aspect='equal')
    im = plt.imshow(a, cmap=cmap, norm=norm, interpolation='none', aspect='equal')
    LW = 3
    if yes:
        def rect(pos):
            r = plt.Rectangle(pos-0.5, 1,1, facecolor="none", edgecolor=EDGE_COL, linewidth=LW)
            plt.gca().add_patch(r)

        x,y = np.meshgrid(np.arange(a.shape[1]),np.arange(a.shape[0]))
        m = np.c_[x[a.astype(bool)],y[a.astype(bool)]]
        for pos in m:
            rect(pos)

    else:
        def rect(pos):
            r = plt.Rectangle(pos-0.5, 2,2, facecolor=COALESCE_COL, edgecolor=EDGE_COL, linewidth=LW)
            plt.gca().add_patch(r)

        x,y = np.meshgrid(np.arange(a.shape[1]),np.arange(a.shape[0]))
        m = np.c_[x[a.astype(bool)],y[a.astype(bool)]]
        print(m)
        m = np.array([[4, 1], [1,4]])
        for pos in m:
            rect(pos)
        pos = np.array((3, 2))
        r = plt.Rectangle(pos-0.5, 1,1, facecolor="none", edgecolor=EDGE_COL, linewidth=LW)
        plt.gca().add_patch(r)
        pos = np.array((5, 5))
        r = plt.Rectangle(pos-0.5, 1,1, facecolor="none", edgecolor=EDGE_COL, linewidth=LW)
        plt.gca().add_patch(r)
        pos = np.array((4, 4))
        r = plt.Rectangle(pos-0.5, 2,1, facecolor=COALESCE_COL, edgecolor=EDGE_COL, linewidth=LW)
        
        plt.gca().add_patch(r)


    from matplotlib.lines import Line2D
    custom_lines = []
    names = []
    names = ['Road', 'House']
    i = -1
    nnn = []
    for n, c in zip(names[::-1], list_of_colours[::-1]):
        i += 1
        # if i != yes: continue
        if not yes: continue
        custom_lines.append(Line2D([0], [0], color=c, lw=4))
        nnn.append(n)
    plt.tight_layout()
    fig = plt.gcf()
    if yes:
        l = plt.legend(custom_lines, nnn, ncol=2, bbox_to_anchor=(0, 0.9, 1, 0.1),
                       loc='upper left', mode='expand', borderaxespad=0)
        for i in l.get_lines(): i.set_linewidth(10)
    remove_axis_ticks_keep_border(plt.gca())

    # plt.show()
    fname = f'plots/paper/coalesce_{yes}.jpg'
    mysavefig(fname, save_pdf=True)
    plt.close()
    
    
if __name__ == '__main__':
    # Plot coalesce vs not
    main(True)
    main(False)