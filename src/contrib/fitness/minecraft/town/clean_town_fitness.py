import numpy as np
from contrib.fitness.common.fitness_utils import get_counts_of_array
from games.level import Level
from games.minecraft.twod.town import Minecraft2DTownLevel
from novelty_neat.fitness.fitness import IndependentNeatFitnessFunction
import skimage.morphology as morph
import scipy.ndimage as nd

class CleanTownReachableFitness(IndependentNeatFitnessFunction):
    # This is a much simpler version of a reachable town fitness, also uses 

    def calc_fitness_single_level(self, level: Level) -> float:
        M = level.map
        assert M.shape[1] == 1, f"BAD: {M.shape}"
        M = M[:, 0, :]
        # See if all 
        ROAD  = level.tile_types_reversed['road']
        HOUSE1 = level.tile_types_reversed['house_1']
        HOUSE3 = level.tile_types_reversed['house_3']
        
        where_is_house = np.logical_and(M >= HOUSE1, M <= HOUSE3)
        new = np.copy(M)
        new[where_is_house] = ROAD
        where_is_road = new == ROAD
        new[~where_is_road] = 0
        new[where_is_road] = 1
        
        labelled_road = morph.label((M == ROAD), connectivity=1)
        
        labelled_connected = morph.label(new, connectivity=1)

        counts2 = get_counts_of_array(labelled_road)
        counts1 = get_counts_of_array(labelled_connected)
        
        diff_1 = np.clip(abs(1 - len(counts1)), 0, 10)
        diff_2 = np.clip(abs(1 - len(counts2)), 0, 10)
        
        return (1 / (diff_1 + 1) + 1 / (diff_2 + 1)) / 2.0

class CleanTownReachableGardensTooFitness(IndependentNeatFitnessFunction):
    # This is a much simpler version of a reachable town fitness, also uses 

    def calc_fitness_single_level(self, level: Level) -> float:
        M = level.map
        assert M.shape[1] == 1, f"BAD: {M.shape}"
        M = M[:, 0, :]
        # See if all 
        ROAD  = level.tile_types_reversed['road']
        HOUSE1 = level.tile_types_reversed['house_1']
        HOUSE3 = level.tile_types_reversed['house_3']
        GARDEN = level.tile_types_reversed['garden']
        
        where_is_house = np.logical_and(M >= HOUSE1, M <= HOUSE3)
        where_is_garden = M == GARDEN
        new = np.copy(M)
        new[where_is_house] = ROAD
        new[where_is_garden] = ROAD
        where_is_road = new == ROAD
        new[~where_is_road] = 0
        new[where_is_road] = 1
        
        labelled_road = morph.label(np.logical_and(M == ROAD, M==GARDEN), connectivity=1)
        
        labelled_connected = morph.label(new, connectivity=1)

        counts2 = get_counts_of_array(labelled_road)
        counts1 = get_counts_of_array(labelled_connected)
        
        diff_1 = np.clip(abs(1 - len(counts1)), 0, 10)
        diff_2 = np.clip(abs(1 - len(counts2)), 0, 10)
        
        return (1 / (diff_1 + 1) + 1 / (diff_2 + 1)) / 2.0
        


class CleanTownHaveRoadsFitness(IndependentNeatFitnessFunction):
    # This is a much simpler version of a reachable town fitness, also uses 
    desired_val = 0.3 # desire 30% of roads
    def calc_fitness_single_level(self, level: Level) -> float:
        M = level.map
        assert M.shape[1] == 1, f"BAD: {M.shape}"
        M = M[:, 0, :]
        # See if all 
        ROAD  = level.tile_types_reversed['road']
        how_many_roads = (M == ROAD).mean()
        
        return np.clip(1 - 11 * (self.desired_val - how_many_roads) ** 2, 0, 1)
    
    
class CleanTownSurroundedBySameTileFitness(IndependentNeatFitnessFunction):
    # This is a much simpler version of a reachable town fitness, also uses 

    def calc_fitness_single_level(self, level: Level) -> float:
        M = level.map
        assert M.shape[1] == 1, f"BAD: {M.shape}"
        M = M[:, 0, :]
        boundary = np.zeros_like(M, dtype=np.bool8)
        for k in [0, -1]:
            boundary[k, :] = 1
            boundary[:, k] = 1
        
        
        counts = get_counts_of_array(M[boundary], ignore_zero=False)
        # I actually want this to be an array of length 1.
        tot = sum([c ** 2 for c in counts])
        
        return tot / M.size ** 2
    
class CleanTownCorrectSurroundedBySameTileFitness(IndependentNeatFitnessFunction):
    # This is a much simpler version of a reachable town fitness, also uses 

    def calc_fitness_single_level(self, level: Level) -> float:
        M = level.map
        assert M.shape[1] == 1, f"BAD: {M.shape}"
        M = M[:, 0, :]
        boundary = np.zeros_like(M, dtype=np.bool8)
        for k in [0, -1]:
            boundary[k, :] = 1
            boundary[:, k] = 1
        
        
        counts = get_counts_of_array(M[boundary], ignore_zero=False)
        # I actually want this to be an array of length 1.
        tot = sum([c ** 2 for c in counts])
        
        return tot / boundary.sum() ** 2
    
    
class CleanTownReachableByRoadsFitness(IndependentNeatFitnessFunction):
    # This is also a reachable fitness, but tries to ensure that each house is reachable via roads, to prevent massive blocks of contiguous houses.
    # Should be used in conjunction with the reachable fitness.
    def calc_fitness_single_level(self, level: Level) -> float:
        M = level.map
        assert M.shape[1] == 1, f"BAD: {M.shape}"
        M = M[:, 0, :]
        # See if all 
        ROAD  = level.tile_types_reversed['road']
        HOUSE1 = level.tile_types_reversed['house_1']
        HOUSE3 = level.tile_types_reversed['house_3']
        filt = [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ]
        test = np.zeros_like(M, dtype=np.int32)
        where_is_house = np.logical_and(M >= HOUSE1, M <= HOUSE3)
        if where_is_house.sum() == 0: return 0
        test[where_is_house] = 10
        test[M == ROAD] = 1
        averages = nd.convolve(test, filt)
        is_bad = (averages % 10 == 0) # 
        return 1 - is_bad[where_is_house].mean()

class CleanTownReachableByRoadsVariableFitness(CleanTownReachableByRoadsFitness):
    # This is also a reachable fitness, but tries to ensure that each house is reachable via roads, to prevent massive blocks of contiguous houses.
    # Should be used in conjunction with the reachable fitness.
    # This says that the optimal percentage is 80%
    def calc_fitness_single_level(self, level: Level) -> float:
        good = super().calc_fitness_single_level(level)
        return np.clip(1 - 20 * (0.8 - good) ** 2, 0, 1)
        

class CleanTownContainsAllHousesFitness(IndependentNeatFitnessFunction):
    # Ensures that these have different types of houses
    def calc_fitness_single_level(self, level: Level) -> float:
        M = level.map
        assert M.shape[1] == 1, f"BAD: {M.shape}"
        M = M[:, 0, :]
        # See if all 
        HOUSE1 = level.tile_types_reversed['house_1']
        HOUSE3 = level.tile_types_reversed['house_3']
        where_is_house = np.logical_and(M >= HOUSE1, M <= HOUSE3)
        new_m = np.copy(M)
        new_m[~where_is_house] = 0
        
        counts = get_counts_of_array(new_m)
        total = sum(counts)
        if total == 0: return 0
        return min(counts) / total / 3        


    
class CleanTownReachableByRoadsAndNormalFitness(IndependentNeatFitnessFunction):
    # This is also a reachable fitness, but tries to ensure that each house is reachable via roads, to prevent massive blocks of contiguous houses.
    # Should be used in conjunction with the reachable fitness.
    def calc_fitness_single_level(self, level: Level) -> float:
        def a():
            M = level.map
            assert M.shape[1] == 1, f"BAD: {M.shape}"
            M = M[:, 0, :]
            # See if all 
            ROAD  = level.tile_types_reversed['road']
            HOUSE1 = level.tile_types_reversed['house_1']
            HOUSE3 = level.tile_types_reversed['house_3']
            filt = [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ]
            test = np.zeros_like(M, dtype=np.int32)
            where_is_house = np.logical_and(M >= HOUSE1, M <= HOUSE3)
            if where_is_house.sum() == 0: return 0
            test[where_is_house] = 10
            test[M == ROAD] = 1
            averages = nd.convolve(test, filt)
            is_bad = (averages % 10 == 0) # 
            return 1 - is_bad[where_is_house].mean()
        def b():
            M = level.map
            assert M.shape[1] == 1, f"BAD: {M.shape}"
            M = M[:, 0, :]
            # See if all 
            ROAD  = level.tile_types_reversed['road']
            HOUSE1 = level.tile_types_reversed['house_1']
            HOUSE3 = level.tile_types_reversed['house_3']
            
            where_is_house = np.logical_and(M >= HOUSE1, M <= HOUSE3)
            new = np.copy(M)
            new[where_is_house] = ROAD
            where_is_road = new == ROAD
            new[~where_is_road] = 0
            new[where_is_road] = 1
            
            labelled_road = morph.label((M == ROAD), connectivity=1)
            
            labelled_connected = morph.label(new, connectivity=1)

            counts2 = get_counts_of_array(labelled_road)
            counts1 = get_counts_of_array(labelled_connected)
            
            diff_1 = np.clip(abs(1 - len(counts1)), 0, 10)
            diff_2 = np.clip(abs(1 - len(counts2)), 0, 10)
            
            return (1 / (diff_1 + 1) + 1 / (diff_2 + 1)) / 2.0

        return (a() + b()) / 2

class CleanTownEachHouseReachableByRoadsFitness(IndependentNeatFitnessFunction):
    # Each house should be reachable via each other house.
    def calc_fitness_single_level(self, level: Level) -> float:
        def a():
            M = level.map
            assert M.shape[1] == 1, f"BAD: {M.shape}"
            M = M[:, 0, :]
            # See if all 
            ROAD  = level.tile_types_reversed['road']
            HOUSE1 = level.tile_types_reversed['house']
            HOUSE3 = level.tile_types_reversed['house']
            filt = [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ]
            test = np.zeros_like(M, dtype=np.int32)
            where_is_house = np.logical_and(M >= HOUSE1, M <= HOUSE3)
            if where_is_house.sum() == 0: return 0
            test[where_is_house] = 10
            test[M == ROAD] = 1
            averages = nd.convolve(test, filt, mode='constant')
            
            # Bad are things that are connected via 0 roads or by 4 roads.
            # is_bad = np.logical_or(averages % 10 == 0, averages % 14 == 0) # 
            is_bad = np.logical_or(averages % 10 == 0, averages == 14) # 
            return 1 - is_bad[where_is_house].mean()

        def b():
            M = level.map
            assert M.shape[1] == 1, f"BAD: {M.shape}"
            M = M[:, 0, :]
            # See if all 
            ROAD  = level.tile_types_reversed['road']
            HOUSE1 = level.tile_types_reversed['house']
            HOUSE3 = level.tile_types_reversed['house']
            
            where_is_house = np.logical_and(M >= HOUSE1, M <= HOUSE3)
            new = np.copy(M)
            new[where_is_house] = ROAD
            where_is_road = new == ROAD
            new[~where_is_road] = 0
            new[where_is_road] = 1
            
            labelled_road = morph.label((M == ROAD), connectivity=1)
            
            labelled_connected = morph.label(new, connectivity=1)

            counts2 = get_counts_of_array(labelled_road)
            counts1 = get_counts_of_array(labelled_connected)
            
            diff_1 = np.clip(abs(1 - len(counts1)), 0, 10)
            diff_2 = np.clip(abs(1 - len(counts2)), 0, 10)
            
            return (1 / (diff_1 + 1) + 1 / (diff_2 + 1)) / 2.0
        x, y = a(), b()
        return (x + y) / 2


class CleanTownEachHouseReachableByRoadsStricterFitness(IndependentNeatFitnessFunction):
    # Each house should be reachable via each other house.
    def calc_fitness_single_level(self, level: Level) -> float:
        def a():
            M = level.map
            assert M.shape[1] == 1, f"BAD: {M.shape}"
            M = M[:, 0, :]
            # See if all 
            ROAD  = level.tile_types_reversed['road']
            HOUSE1 = level.tile_types_reversed['house']
            HOUSE3 = level.tile_types_reversed['house']
            filt = [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ]
            test = np.zeros_like(M, dtype=np.int32)
            where_is_house = np.logical_and(M >= HOUSE1, M <= HOUSE3)
            if where_is_house.sum() == 0: return 0
            test[where_is_house] = 10
            test[M == ROAD] = 1
            averages = nd.convolve(test, filt, mode='constant')
            
            # Bad are things that are connected via 0 roads or by 4 roads.
            # is_bad = np.logical_or(averages % 10 == 0, averages % 14 == 0) # 
            is_bad = np.logical_or(averages % 10 == 0, averages == 14) # 
            
            temp = np.clip(is_bad[where_is_house].sum() / 20, 0, 1)
            ans = 1 - temp
            ans = ans ** 2
            return ans

        def b():
            M = level.map
            assert M.shape[1] == 1, f"BAD: {M.shape}"
            M = M[:, 0, :]
            # See if all 
            ROAD  = level.tile_types_reversed['road']
            HOUSE1 = level.tile_types_reversed['house']
            HOUSE3 = level.tile_types_reversed['house']
            
            where_is_house = np.logical_and(M >= HOUSE1, M <= HOUSE3)
            new = np.copy(M)
            new[where_is_house] = ROAD
            where_is_road = new == ROAD
            new[~where_is_road] = 0
            new[where_is_road] = 1
            
            labelled_road = morph.label((M == ROAD), connectivity=1)
            labelled_connected = morph.label(new, connectivity=1)

            counts2 = get_counts_of_array(labelled_road)
            counts1 = get_counts_of_array(labelled_connected)
            diff_1 = np.clip(abs(1 - len(counts1)), 0, 10)
            diff_2 = np.clip(abs(1 - len(counts2)), 0, 10)
            t = (1 / (diff_1 + 1) * 1 / (diff_2 + 1)) / 1.0
            return t
            
        x, y = a(), b()
        return x * y

if __name__ == '__main__':
    level = Minecraft2DTownLevel()
    r=1
    h=2
    p=0
    level.map = np.array([
        [r, r, r, r, r, r, r, r, r, r],
        [r, h, r, r, r, r, r, r, r, r],
        [r, h, r, r, r, r, r, p, p, r],
        [r, h, r, r, r, r, r, p, p, r],
        [r, h, r, r, h, r, r, p, p, r],
        [r, h, r, r, r, r, r, p, p, r],
        [r, h, r, r, h, r, h, p, h, r],
        [r, h, r, h, p, h, h, p, h, r],
        [r, r, r, r, h, r, h, h, h, r],
        [r, r, r, r, r, r, r, r, r, r],
    ]).reshape(10, 1, 10)
    
    # h, p = r, r
    # level.map = np.array([
    #     [r, r, r, r, r, r, r, r, r, r],
    #     [r, h, r, r, r, r, r, r, r, r],
    #     [r, h, r, r, r, r, r, p, p, r],
    #     [r, h, r, r, r, r, r, p, p, r],
    #     [r, h, r, r, r, r, r, p, p, r],
    #     [r, h, r, r, r, r, r, p, p, r],
    #     [r, h, r, r, r, r, h, h, h, r],
    #     [r, h, r, r, r, r, h, h, h, r],
    #     [r, r, r, r, r, r, h, h, h, r],
    #     [r, r, r, r, r, r, r, r, r, r],
    # ]).reshape(10, 1, 10)
    
    # fit = CleanTownReachableByRoadsFitness()
    # print("FIT", fit.calc_fitness_single_level(level))    
    fit = CleanTownEachHouseReachableByRoadsFitness()
    print("FIT", fit.calc_fitness_single_level(level))