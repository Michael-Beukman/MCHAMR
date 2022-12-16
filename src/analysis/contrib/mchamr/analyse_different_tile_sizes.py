
from collections import defaultdict
import glob

from matplotlib import pyplot as plt
import numpy as np
from common.utils import mysavefig, plot_mean_std
from contrib.fitness.common.aggregate_level_fitness import AggregateLevelFitness
from contrib.fitness.minecraft.town.clean_town_fitness import CleanTownEachHouseReachableByRoadsStricterFitness
from games.minecraft.twod.town import Minecraft2DTownLevel
from runs.proper_experiments.group.utils.game_utils import _glob_get_first, _glob_get_latest, get_generator_net_game_from_pickle, get_prob_fit_function
import seaborn as sns

sns.set_theme()
K = set()
GOODS = {
    'b':r'$1\times 1$',
    'c':r'$2\times 2$',
    'd':r'$3\times 3$',
    'e':r'$4\times 4$',
    'f':r'$5\times 5$',
    'g':r'$10\times 10$',
}
def main(experiment_name, keys_to_use=[], better_main=[]):
    global K
    directory = _glob_get_first(f'../results/experiments/pcgnn_{experiment_name}/*/PCGNN/')
    
    directory = _glob_get_latest(f'{directory}/*')
    test = defaultdict(lambda: [])
    for seed in range(10):
        p = _glob_get_first(f'{directory}/*/*/{seed}/*.pbz2')
        generator, net, game, dic = get_generator_net_game_from_pickle(p, return_entire_dic=True)
        
        alls = dic['train_results'][0]['all_genomes_fitnesses']
        d = defaultdict(lambda: [])
        K |= set((alls)[0]['fitness'].keys())
        keys = keys_to_use
        better = {k: b for k, b in zip(keys, better_main)}
        for i in range(len(alls)):
            for k, v in (alls)[i]['fitness'].items():
                d[k].append(v[-1])
        for k in keys:
            v = d[k]
            test[k].append(v)
    
    for k, v in test.items():
        m = np.mean(v, axis=0)
        s = np.std(v, axis=0)
        A = GOODS.get(experiment_name.split("-")[-1][-1], experiment_name)
        plot_mean_std(m, s, label=f"{A}")

def compare_tile_sizes():
    all_keys = ['AGG[CleanTownEachHouseReachableByRoadsStricterFitness; 1]', 'AGG[Minecraft2DTownHasHousesAndGardensFitness; 1]', 'NoveltyIntraGenerator', 'AGG[ProbabilityDistributionFitness; 1]', 'all', 'NoveltyMetric', 'AGG[CleanTownEachHouseReachableByRoadsFitness; 1]']
    better = ['StrictReach', 'EqualDist', 'IntraNovelty', 'Prob', 'all', 'NoveltyMetric', 'Reachable']
    clean = ['Reachability', 'Entropy', 'Intra-Novelty', 'Probability Fitness', 'Total', 'Novelty Metric', 'Reachability']
    for k, bet, bet2 in zip(all_keys, better, clean):
        for T in ['3551-hl']:
            try:
                for let in ['b', 'c', 'd', 'e', 'f', 'g']:
                    main(f'{T}{let}', keys_to_use=[k], better_main=[bet])
            except Exception as e:
                print('failed', T, e)
                plt.close()
                continue
            plt.xlabel("Generations")
            plt.ylabel(f"Fitness - {bet2}")
            plt.legend()
            plt.tight_layout()
            mysavefig(n := f'../results/analysis/hierarchy/{T}_{bet}.jpg', dpi=400, save_pdf=True)
            print('saved to ', n)
            plt.close()


def plot_sample_levels(agg):
    from runs.proper_experiments.group.mcc.eval.plot_utils import PLOT_DIR, plot_single_level
    do_aggregate = agg
    sns.set_theme()
    # seed = 0
    np.random.seed(42)
    fit_prob = get_prob_fit_function({'house': 0.4, 'garden': 0.3, 'road': 0.3}, level_class=Minecraft2DTownLevel)(1, None)
    fit_reach = CleanTownEachHouseReachableByRoadsStricterFitness()
    for seed in range(3):
        fig, axs = plt.subplots(2, 3, figsize=(3*5, 2*5))
        for i, (let, ax) in enumerate(zip(['b', 'c', 'd', 'e', 'f', 'g'], axs.ravel())):
            print(let)
            experiment_name = f'3551-hl{let}'
            directory = _glob_get_first(f'../results/experiments/pcgnn_{experiment_name}/*/PCGNN/')
            directory = _glob_get_latest(f'{directory}/*')
            p = _glob_get_first(f'{directory}/*/*/{seed}/*.pbz2')
            generator, net, game, dic = get_generator_net_game_from_pickle(p, return_entire_dic=True)
            level = generator(net)

            _s = level.map.shape[0]
            
            a = AggregateLevelFitness(fitnesses=[], weights=[], tile_size=_s // 10, default_tile=0)
            level2 = a.get_aggregate_level(level)
            a = fit_prob.calc_fitness_single_level(level2)
            b = fit_reach.calc_fitness_single_level(level2)
            if agg:
                level = level2

            plot_single_level(level, ax, annot=False)
            ax.set_title(f'{GOODS[let]} | Fitness = {np.round((a + b)/2, 2)}', fontsize=24)
        mysavefig(f'../results/analysis/hierarchy/examples/example_levels_{seed}_{agg}.jpg', dpi=400, save_pdf=True)
        plt.close()


if __name__ == '__main__':
    compare_tile_sizes()
    plot_sample_levels(True)
    plot_sample_levels(False)