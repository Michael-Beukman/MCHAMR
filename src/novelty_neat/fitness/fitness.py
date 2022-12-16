
from typing import Any, Dict, List, final
from experiments.logger import Logger, NoneLogger
import numpy as np
from games.level import Level
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from novelty_neat.generation import NeatLevelGenerator
from novelty_neat.maze.neat_maze_level_generation import GenerateMazeLevelsUsingTiling

from novelty_neat.types import LevelNeuralNet


class NeatFitnessFunction:
    """
        A general fitness function interface. Subclasses should implement calc_fitness.
    """

    def __init__(self, number_of_levels_to_generate: int = 5, level_gen: NeatLevelGenerator = None, logger: Logger = NoneLogger()):
        """
        Args:
            number_of_levels_to_generate (int, optional): The number of levels to generate per network. Defaults to 5.
            level_gen (NeatLevelGenerator, optional): The way to generate levels. Defaults to None.
        """
        self.number_of_levels = number_of_levels_to_generate
        self.level_gen = level_gen
        self.steps = -1
        self.logger = logger

    def get_levels(self, nets: List[LevelNeuralNet]) -> List[List[Level]]:
        """Returns a list of list of levels from these neural nets. The shape of this list is (len(nets), self.number_of_levels)

        Args:
            nets (List[LevelNeuralNet]): [description]

        Returns:
            list[Level]: [description]
        """
        levels = []
        for net in nets:
            temp = []
            for i in range(self.number_of_levels):
                temp.append(self.level_gen(net))
            levels.append(temp)
        return levels

    def calc_fitness(self, nets: List[LevelNeuralNet], levels: List[List[Level]]) -> List[float]:
        """Should calculate the fitness from these levels.

        Args:
            nets (List[LevelNeuralNet]): These networks generated these levels. Ideally you won't use these, but for some use-cases, like the Novelty Metric, this is crucial.
            levels (List[List[Level]]): levels[i][j] corresponds to network i, level j.

        Returns:
            List[float]: Returns a list with the same length as levels representing the fitnesses for each network.
        """
        raise NotImplementedError()

    @final
    def __call__(self, nets: List[LevelNeuralNet]) -> List[float]:
        levels = self.get_levels(nets)
        self.steps += 1
        return self.calc_fitness(nets, levels)

    def params(self) -> Dict[str, Any]:
        """Returns a dictionary representing the fitness function parameters, and specifications.

        Returns:
            Dict[str, Any]
        """
        return {
            'name': self.__class__.__name__,
            'number_of_levels': self.number_of_levels,
            'level_generator': type(self.level_gen).__name__,
            'repr': repr(self)
        }
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.number_of_levels}, {self.level_gen})"
    
    def reset(self) -> None:
        """
            Resets any variables in the class (like the novelty archive)
        """
        pass
    
    def name(self) -> str:
        return self.__class__.__name__


class CombinedFitness(NeatFitnessFunction):
    """Combines multiple fitness measures with a weighted sum.
    """

    def __init__(self, fitnesses: List[NeatFitnessFunction], weights: List[float], number_of_levels_to_generate: int, level_gen: NeatLevelGenerator, mode: str = 'add'):
        """
        Args:
            fitnesses (List[NeatFitnessFunction]): The fitness metrics to use.
            weights (List[float]): The weights. This will be normalised to sum to one.
            number_of_levels_to_generate (int): Number of levels to generate to calculate the fitness
            level_gen (NeatLevelGenerator): How to generate levels.
            mode (str): If 'add', then averages the fitnesses, otherwise multiplies them
        """
        super().__init__(number_of_levels_to_generate, level_gen)
        self.weights: np.ndarray = np.array(weights) / sum(weights)
        self.fitnesses = fitnesses
        assert len(self.fitnesses) == len(self.weights)
        self.mode = mode
        self.fitness_components = []

    def calc_fitness(self, nets: List[LevelNeuralNet], levels: List[List[Level]]) -> List[float]:
        self.fitness_components = []
        final_answer = np.array([0.0 for _ in levels])
        if (self.mode == 'mult'): final_answer += 1
        dic = {
            'CombinedFitness': {
                n.name(): {
                    'mean': 0,
                    'max': 0,
                    'min': 0,
                    'all': []
                } for n in self.fitnesses
            }
        }
        # best individual's different components
        best_indiv_components = {}
        new_list_of_all_things = {}
        for func, weight in zip(self.fitnesses, self.weights):
            list_of_all = np.array(func.calc_fitness(nets, levels))
            new_list_of_all_things[func.name()] = list_of_all
            k = dic['CombinedFitness'][func.name()]
            k['mean'] = np.mean(list_of_all)
            k['max'] = np.max(list_of_all)
            k['min'] = np.min(list_of_all)
            
            k['all'] = (list_of_all)
            if self.mode == 'add':
                final_answer += weight * list_of_all
            else:
                final_answer *= list_of_all
        best_index = np.argmax(final_answer)
        for key in new_list_of_all_things:
            best_indiv_components[key] = new_list_of_all_things[key][best_index]
        dic['BestIndivFitness'] = {
            'total': final_answer[best_index],
            'components': best_indiv_components
        }
        self.fitness_components = [
            {key: val['all'][i] for key, val in dic['CombinedFitness'].items()} for i in range(len(levels))
        ]
        if not (self.logger and self.logger.LOG_ALL_FITNESSES): # clear these items if necessary
            for key, v in dic['CombinedFitness'].items():
                v['all'] = []
        self.logger.log(dic, step=self.steps)
        return final_answer

    def params(self) -> Dict[str, Any]:
        dic = super().params()
        dic['components'] = []
        for func in self.fitnesses:
            dic['components'].append(func.params())
        dic['weights'] = list(self.weights)
        return dic

    def __repr__(self) -> str:
        return f"CombinedFitness({self.fitnesses}, {self.weights}, {self.number_of_levels}, {self.level_gen}, mode={self.mode})"

    def reset(self) -> None:
        for func in self.fitnesses:
            func.reset()
        
class SimpleNeatFitnessFunction(NeatFitnessFunction):
    """A fitness function that allows subclasses to only implement a function that takes in a list of list of levels and returns a list of floats. Must implement `calc_fitness_for_levels`
    """
    def __init__(self, number_of_levels_to_generate: int = 5, level_gen: NeatLevelGenerator = None, logger: Logger = NoneLogger()):
        super().__init__(number_of_levels_to_generate, level_gen, logger)
    
    @final
    def calc_fitness(self, nets: List[LevelNeuralNet], levels: List[List[Level]]) -> List[float]:
        assert len(levels[0]) == self.number_of_levels
        return self.calc_fitness_for_levels(levels)
    
    def calc_fitness_for_levels(self, levels: List[List[Level]]) -> List[float]:
        raise NotImplementedError()
    

class IndependentNeatFitnessFunction(SimpleNeatFitnessFunction):
    """Assumes each level's fitness is independent of the other levels -- makes it easy to work in this scenario.
            Simple implement `calc_fitness_single_level` that returns a single float for a single level.
    """
    def __init__(self, number_of_levels_to_generate: int = 5, level_gen: NeatLevelGenerator = None, logger: Logger = NoneLogger()):
        super().__init__(number_of_levels_to_generate, level_gen, logger)
    
    @final
    def calc_fitness_for_levels(self, levels: List[List[Level]]) -> List[float]:
        final_answer = []
        for group in levels:
            d = 0
            for l in group:
                d += self.calc_fitness_single_level(l)
            final_answer.append(d / len(group))
            assert 0 <= final_answer[-1] <= 1, f"Incorrect values, {final_answer[-1]}"
        
        return final_answer
    
    def calc_fitness_single_level(self, level: Level) -> float:
        raise NotImplementedError()


if __name__ == '__main__':
    func = CombinedFitness([], [1], 5, GenerateMazeLevelsUsingTiling(game=MazeGame(MazeLevel())))
    print(func.params())
    pass