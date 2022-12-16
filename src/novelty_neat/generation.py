from typing import Tuple, Union
import numpy as np
from games.game import Game
from games.level import Level
from novelty_neat.types import LevelNeuralNet


class NeatLevelGenerator:
    """A superclass for generating levels in different ways.
    """
    def __init__(self, number_of_random_variables: int = 2):
        """
        Args:
            number_of_random_variables (int, optional): The number of random inputs the neural net takes. Defaults to 2.
        """
        self.number_of_random_variables = number_of_random_variables
    
    def __call__(self, net: LevelNeuralNet, input: Union[np.ndarray, None] = None) -> Level:
        """Generates a level from the given network and the input. 

        Args:
            net (LevelNeuralNet): [description]
            input (Union[np.ndarray, None], optional): What to give to the network. If input is None, then we randomly create an input 
                of length self.number_of_random_variables
                Defaults to None.
        Returns:
            Level: [description]
        """
        if input is None: 
            input = np.random.randn(self.number_of_random_variables)
        return self.generate_level(net, input)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.number_of_random_variables})"

    def generate_level(self, net: LevelNeuralNet, input: np.ndarray) -> Level:
        """This should actually generate the level.

        Args:
            net (LevelNeuralNet):
            input (np.ndarray):
        
        Returns:
            Level:
        """
        raise NotImplementedError('')
    
    def get_input_output_size(self) -> Tuple[int, int]:
        """This should return the required input and output size for a network.
             For instance, if this method gives the network a vector of length 12 and expects an output of size 14, then this function should return [12, 14]
        Returns:
            Tuple[int, int]: 
        """
        raise NotImplementedError()
    
class SeedableLevelGenerator(NeatLevelGenerator):
    game: Game
    def generate_level(self, net: LevelNeuralNet, input: np.ndarray, start_output: np.ndarray = None) -> Level:
        """Generates a level using this network and the given input. The `start_output`, however specifies the starting level, instead of being random.

        Args:
            net (LevelNeuralNet): [description]
            input (Union[np.ndarray, None], optional): What to give to the network. If input is None, then we randomly create an input 
                of length self.number_of_random_variables
                Defaults to None.
            start_output (np.ndarray): level.map

        Returns:
            Level: 
        """
        raise NotImplementedError()