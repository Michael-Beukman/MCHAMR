import glob
import fire
from matplotlib import pyplot as plt
import numpy as np
from contrib.fitness.common.aggregate_level_fitness import AggregateLevelFitness

from common.utils import mysavefig, remove_axis_ticks_keep_border
from runs.proper_experiments.group.mcc.eval.plot_utils import PLOT_DIR, plot_single_level
from runs.proper_experiments.group.utils.game_utils import _glob_get_first, _glob_get_latest, get_generator_net_game_from_pickle
    
def main(experiment_name: str,
         max_seeds: int = 10,
         max_runs_per_seed: int = 5,
         level_size: int = 10,
         do_aggregate=False
         ):
    agg_name = '' if not do_aggregate else '_agg'
    np.random.seed(42)
    directory = _glob_get_first(f'../results/experiments/pcgnn_{experiment_name}/*/PCGNN/')
    
    directory = _glob_get_latest(f'{directory}/*')
    total_seeds = len(glob.glob(f'{directory}/*/*/*/*.pbz2'))
    seeds_to_iterate = min(total_seeds, max_seeds)
    _size = 5
    fig, all_axs = plt.subplots(seeds_to_iterate, max_runs_per_seed, figsize=(_size*max_runs_per_seed, _size*seeds_to_iterate+0))

    all_axs = all_axs.reshape(seeds_to_iterate, max_runs_per_seed)
    all_solvs = []
    for seed, axs in zip(range(seeds_to_iterate), all_axs):
        p = _glob_get_first(f'{directory}/*/*/{seed}/*.pbz2')
        generator, net, game, dic = get_generator_net_game_from_pickle(p, return_entire_dic=True)
        solv = dic.get('eval_results_single', {}).get('SolvabilityMetric', None)
        solv_string = '' if solv is None else f' - Solv {solv}'
        all_solvs.append(solv)
        for i, ax in enumerate(axs):
            level = generator(net)
            if do_aggregate:
                _s = level.map.shape[0]
                
                a = AggregateLevelFitness(fitnesses=[], weights=[], tile_size=_s // 10, default_tile=0)
                level = a.get_aggregate_level(level)

            SS = ''
            plot_single_level(level, ax, annot=False if level.map.size >= 2500 else True)
            ax.set_title(f"Seed {seed}{solv_string}-{SS}")
            remove_axis_ticks_keep_border(ax)

    solv_str = '' if all([s is None for s in all_solvs]) else f' -- Avg Solv {np.round(np.mean(all_solvs), 2)}'
    plt.suptitle(f"Levels for {experiment_name}{solv_str}", fontsize=48)
    fname = f'{PLOT_DIR}/group/{experiment_name.split("-")[0]}/{experiment_name}/levels{agg_name}.jpg'
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    mysavefig(fname, do_tight_layout=False)
    print(f"Saved to {fname}")

if __name__ == '__main__':
    # Usage is like ./run.sh src/runs/proper_experiments/group/mcc/eval/viz2d.py v03351-hlb
    fire.Fire(main)