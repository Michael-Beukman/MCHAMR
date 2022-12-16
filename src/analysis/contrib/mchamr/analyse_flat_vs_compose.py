import copy
import glob
import os

from matplotlib import pyplot as plt
import numpy as np
import ray
from common.utils import mysavefig, plot_mean_std, save_compressed_pickle
from games.minecraft.twod.compose.flat_town import Flat2DTownLevel
from runs.proper_experiments.group.utils.game_utils import _glob_get_first, _glob_get_latest, get_generator_net_game_from_pickle
import seaborn as sns
import neat
from contrib.fitness.minecraft.twod.composed.random_layouts_fitness import Minecraft2DFlatTownRandomLayoutFitness
from common.utils import load_compressed_pickle, mysavefig
from runs.proper_experiments.group.mcc.eval.plot_utils import plot_single_level

sns.set_theme()
def get_gen(experiment_name, seed):
    directory = _glob_get_first(f'../results/experiments/pcgnn_{experiment_name}/*/PCGNN/')
    
    directory = _glob_get_latest(f'{directory}/*')
    p = _glob_get_first(f'{directory}/*/*/{seed}/*.pbz2')
    generator, net, game, dic = get_generator_net_game_from_pickle(p, return_entire_dic=True)
    return generator, net, game, dic

def get_composed_level_from_nets(town_gen, town_net, house_gen, house_net):
    # level_town = ans_town[0](ans_town[1])
    level_town = town_gen(town_net)
    big = np.zeros((25, 25))
    for i in range(level_town.map.shape[0]):
        for j in range(level_town.map.shape[-1]):
            if level_town.map[i, 0, j] == level_town.tile_types_reversed['house']:
                t = house_gen(house_net)
                assert t.map.shape[1] == 1
                T = t.map[:, 0, :]
                D = T.copy()
                T[D == 1] = 2
                T[D == 0] = 1
                big[i*5:i*5+5, j*5:j*5+5] = T
            else:
                big[i*5:i*5+5, j*5:j*5+5] = 0
    return Flat2DTownLevel.from_map(big[:, None])

from tqdm import trange
def do_compositional(OPTION_NUMBER = 0):
    # Compare fitness vs not .
    # Need to compose the things.
    @ray.remote
    def get_fits(experiment_name_flat, experiment_name_town, experiment_name_house, index):
        my_llll_compose = []
        my_llll_flat = []
        F = Minecraft2DFlatTownRandomLayoutFitness(index, 1, None)
        fits = np.zeros((SEEDS := 10, 150))
        fits_flat = np.zeros((SEEDS := 10, 150))
        for seed in range(SEEDS):
            print(experiment_name_flat, seed)
            ans_town   = get_gen(experiment_name_town, seed)
            ans_house  = get_gen(experiment_name_house, seed)
            
            ans_flat  = get_gen(experiment_name_flat, seed)
            
            print("")
            for i in trange(0, 150):
                # print(f"\r{i}     ")
                alls_town = ans_town[-1]['train_results'][0]['all_genomes_fitnesses']
                alls_house = ans_house[-1]['train_results'][0]['all_genomes_fitnesses']
                alls_flat = ans_flat[-1]['train_results'][0]['all_genomes_fitnesses']
                
                town_net = (alls_town[i]['genomes'][-1])[-1]
                house_net = (alls_house[i]['genomes'][-1])[-1]
                flat_net = (alls_flat[i]['genomes'][-1])[-1]
                town_net = neat.nn.FeedForwardNetwork.create(town_net, ans_town[-1]['extra_results']['entire_method'].neat_config)
                house_net = neat.nn.FeedForwardNetwork.create(house_net, ans_house[-1]['extra_results']['entire_method'].neat_config)
                flat_net = neat.nn.FeedForwardNetwork.create(flat_net, ans_flat[-1]['extra_results']['entire_method'].neat_config)
                
                tmp = []
                tmp2 = []
                for p in range(5):
                    level = get_composed_level_from_nets(ans_town[0], town_net, ans_house[0], house_net)
                    tmp.append(F.calc_fitness_single_level(level))
                    
                    my_llll_compose.append(copy.deepcopy(level))
                    
                    level = ans_flat[0](flat_net)
                    
                    my_llll_flat.append(copy.deepcopy(level))
                    
                    tmp2.append(F.calc_fitness_single_level(level))
                fits[seed, i] = np.mean(tmp)
                fits_flat[seed, i] = np.mean(tmp2)
        return fits, fits_flat, my_llll_compose, my_llll_flat
    
    all_compose = []
    all_flat = []
    all_names = [ 'aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at',]
    
    if OPTION_NUMBER == 0:
        all_vals = ray.get(
            [get_fits.remote(f'3558-{name}', f'3559-{name}', '3555-zz', index=index) for index, name in enumerate(all_names)]
        )
    elif OPTION_NUMBER == 1:
        all_vals = ray.get(
            [get_fits.remote(f'3560-{name}', f'3557-{name}', '3555-zzz', index=index) for index, name in enumerate(all_names)]
        )
    all_levels_flat = []
    all_levels_compose = []
    for fc, ff, my_llll_compose, my_llll_flat in all_vals:
        all_flat.append(ff)
        all_compose.append(fc)
        all_levels_flat.append(my_llll_flat)
        all_levels_compose.append(my_llll_compose)
    
    
    all_flat = np.array(all_flat)
    all_compose = np.array(all_compose)
    
    DIR = f"../results/analysis/flat_vs_compose/{OPTION_NUMBER}"
    os.makedirs(DIR, exist_ok=True)
    # OPTION_NUMBER
    all_dics = {'all_flat': all_flat, 'all_compose': all_compose, 'all_levels_compose': all_levels_compose, 'all_levels_flat': all_levels_flat}
    save_compressed_pickle(f"{DIR}/all_dics", all_dics)
    print(all_flat.shape, all_compose.shape)
    all_compose = np.mean(all_compose, axis=1)
    all_flat = np.mean(all_flat, axis=1)
    fits = all_compose
    fits_flat = all_flat
    
    plot_mean_std(np.mean(fits, axis=0), np.std(fits, axis=0), label='Composed')
    plot_mean_std(np.mean(fits_flat, axis=0), np.std(fits_flat, axis=0), label='Flat')
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.legend()
    mysavefig(f"{DIR}/composed_vs_not.jpg", save_pdf=True)

def plot_pretty_levels():
    from contrib.fitness.minecraft.twod.composed import layouts
    W = 2
    A = 1
    R = 0
    plt.figure(figsize=(20, 20))
    index_to_use = 3
    town = layouts.DATA[index_to_use]
    assert town.shape == (5, 5)
    perfect_house = np.array([
        [W, W, W, W, W],
        [W, A, A, A, W],
        [W, A, A, A, W],
        [W, A, A, A, W],
        [W, W, W, W, W],
    ])
    big = np.zeros((25, 25))
    for i in range(town.shape[0]):
        for j in range(town.shape[-1]):
            if town[i, j] == 1: # This is a house
                big[i*5:(i+1)*5, j*5:(j+1)*5] = perfect_house.copy()
            else:
                assert town[i, j] == 0 # is a road
                big[i*5:(i+1)*5, j*5:(j+1)*5] = np.zeros_like(perfect_house) * R
    L = Flat2DTownLevel()
    big = big[:, None]
    L.map = (big)    
    DIR = f"../results/analysis/flat_vs_compose/examples"
    plot_single_level(L, plt.gca(), annot=False, rasterized=True)
    mysavefig(f"{DIR}/correct_level.jpg", save_pdf=True)


def plot_more_levels():
    test = load_compressed_pickle('../results/analysis/flat_vs_compose/1/all_dics.pbz2')
    DIR = f"../results/analysis/flat_vs_compose/examples"
    compose = ((test['all_levels_compose'][3][-1]))
    flat    = ((test['all_levels_flat'][3][-1]))
    
    plt.figure(figsize=(20, 20))
    plot_single_level(compose, plt.gca(), annot=False, rasterized=True)
    mysavefig(f"{DIR}/compose.jpg", save_pdf=True)
    plt.close()
    
    plt.figure(figsize=(20, 20))
    plot_single_level(flat, plt.gca(), annot=False, rasterized=True)
    mysavefig(f"{DIR}/flat.jpg", save_pdf=True)
    plt.close()
    
    pass

def plot_hand_designed_levels():
    experiment_name_town  = '3554-aa'
    experiment_name_house = '3555-zzz'
    experiment_name_flat  = '3556-be'
    
    DIR = f"../results/analysis/flat_vs_compose/examples2"
    
    for seed in range(1):
        print(experiment_name_flat, seed)
        ans_town   = get_gen(experiment_name_town, seed)
        ans_house  = get_gen(experiment_name_house, seed)
        
        ans_flat  = get_gen(experiment_name_flat, seed)
        
        print("")
        for i in trange(149, 150):
            # print(f"\r{i}     ")
            alls_town = ans_town[-1]['train_results'][0]['all_genomes_fitnesses']
            alls_house = ans_house[-1]['train_results'][0]['all_genomes_fitnesses']
            alls_flat = ans_flat[-1]['train_results'][0]['all_genomes_fitnesses']
            
            town_net = (alls_town[i]['genomes'][-1])[-1]
            house_net = (alls_house[i]['genomes'][-1])[-1]
            flat_net = (alls_flat[i]['genomes'][-1])[-1]
            town_net = neat.nn.FeedForwardNetwork.create(town_net, ans_town[-1]['extra_results']['entire_method'].neat_config)
            house_net = neat.nn.FeedForwardNetwork.create(house_net, ans_house[-1]['extra_results']['entire_method'].neat_config)
            flat_net = neat.nn.FeedForwardNetwork.create(flat_net, ans_flat[-1]['extra_results']['entire_method'].neat_config)
            
            level_composed = get_composed_level_from_nets(ans_town[0], town_net, ans_house[0], house_net)
            level_flat = ans_flat[0](flat_net)

            plt.figure(figsize=(20, 20))
            plot_single_level(level_composed, plt.gca(), annot=False, rasterized=True)
            mysavefig(f"{DIR}/compose.jpg", save_pdf=True)
            plt.close()
            
            plt.figure(figsize=(20, 20))
            plot_single_level(level_flat, plt.gca(), annot=False, rasterized=True)
            mysavefig(f"{DIR}/flat.jpg", save_pdf=True)
            plt.close()

        W = 2
        A = 1
        R = 0
        correct_ans = np.array([
            [W, W, W, W, W, R, R, R, R, R, W, W, W, W, W, R, R, R, R, R, W, W, W, W, W],
            [W, A, A, A, W, R, R, R, R, R, W, A, A, A, W, R, R, R, R, R, W, A, A, A, W],
            [W, A, A, A, W, R, R, R, R, R, W, A, A, A, W, R, R, R, R, R, W, A, A, A, W],
            [W, A, A, A, W, R, R, R, R, R, W, A, A, A, W, R, R, R, R, R, W, A, A, A, W],
            [W, W, W, W, W, R, R, R, R, R, W, W, W, W, W, R, R, R, R, R, W, W, W, W, W],
            [R, R, R, R, R, R, R, R, R, R, W, W, W, W, W, R, R, R, R, R, R, R, R, R, R],
            [R, R, R, R, R, R, R, R, R, R, W, A, A, A, W, R, R, R, R, R, R, R, R, R, R],
            [R, R, R, R, R, R, R, R, R, R, W, A, A, A, W, R, R, R, R, R, R, R, R, R, R],
            [R, R, R, R, R, R, R, R, R, R, W, A, A, A, W, R, R, R, R, R, R, R, R, R, R],
            [R, R, R, R, R, R, R, R, R, R, W, W, W, W, W, R, R, R, R, R, R, R, R, R, R],
            [W, W, W, W, W, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, W, W, W, W, W],
            [W, A, A, A, W, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, W, A, A, A, W],
            [W, A, A, A, W, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, W, A, A, A, W],
            [W, A, A, A, W, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, W, A, A, A, W],
            [W, W, W, W, W, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, W, W, W, W, W],
            [R, R, R, R, R, R, R, R, R, R, W, W, W, W, W, R, R, R, R, R, R, R, R, R, R],
            [R, R, R, R, R, R, R, R, R, R, W, A, A, A, W, R, R, R, R, R, R, R, R, R, R],
            [R, R, R, R, R, R, R, R, R, R, W, A, A, A, W, R, R, R, R, R, R, R, R, R, R],
            [R, R, R, R, R, R, R, R, R, R, W, A, A, A, W, R, R, R, R, R, R, R, R, R, R],
            [R, R, R, R, R, R, R, R, R, R, W, W, W, W, W, R, R, R, R, R, R, R, R, R, R],
            [W, W, W, W, W, R, R, R, R, R, W, W, W, W, W, R, R, R, R, R, W, W, W, W, W],
            [W, A, A, A, W, R, R, R, R, R, W, A, A, A, W, R, R, R, R, R, W, A, A, A, W],
            [W, A, A, A, W, R, R, R, R, R, W, A, A, A, W, R, R, R, R, R, W, A, A, A, W],
            [W, A, A, A, W, R, R, R, R, R, W, A, A, A, W, R, R, R, R, R, W, A, A, A, W],
            [W, W, W, W, W, R, R, R, R, R, W, W, W, W, W, R, R, R, R, R, W, W, W, W, W]])
        correct_ans = Flat2DTownLevel.from_map(correct_ans[:, None])
        plt.figure(figsize=(20, 20))
        plot_single_level(correct_ans, plt.gca(), annot=False, rasterized=True)
        mysavefig(f"{DIR}/correct_level.jpg", save_pdf=True)
        plt.close()

if __name__ == '__main__':
    plot_hand_designed_levels()
    plot_more_levels()
    plot_pretty_levels()
    ray.init()
    do_compositional(0)
    plt.close()
    do_compositional(1)
    plt.close()