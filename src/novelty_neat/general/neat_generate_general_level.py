from typing import List, Tuple
import numpy as np
from games.game import Game
from games.level import Level
from novelty_neat.generation import NeatLevelGenerator, SeedableLevelGenerator
from novelty_neat.types import LevelNeuralNet

def _get_inout_dim(self, DIM):
    one_hot_factor = 1
    
    if self.use_one_hot_encoding:
        one_hot_factor = len(self.game.level.tile_types)
    
    t_s = self.context_size * 2 + 1
    p_s = self.predict_size**DIM
    in_dim = (t_s**DIM - 1) * one_hot_factor + self.number_of_random_variables
    if self.predict_size > 1:
        in_dim = one_hot_factor * (t_s + self.predict_size - 1)**DIM + self.number_of_random_variables
    out_dim = p_s * len(self.game.level.tile_types)
    if hasattr(self, 'input_coords') and self.input_coords:
        in_dim += DIM
    if hasattr(self, 'input_linear_coords') and self.input_linear_coords:
        in_dim += 1
    if hasattr(self, 'add_bias') and self.add_bias:
        in_dim += 1
    return in_dim, out_dim
        

def _one_hot(value: int, total: int) -> List[float]:
    arr = [0] * total
    arr[int(value)] = 1.0
    return arr


class GenerateGeneralLevelUsingTiling(SeedableLevelGenerator):
    def __init__(self, game: Game,
                 context_size: int = 1, number_of_random_variables: int = 2,
                 do_padding_randomly: bool = False,
                 random_perturb_size: float = 0,
                 predict_size: int = 1,
                 reversed_direction: int = 0,
                 use_one_hot_encoding: int = False,
                 input_coords: bool = False,
                 add_bias: bool = False,
                 input_linear_coords: bool = False,
                 normalisation_range_for_coordinates: Tuple[float, float] = (0, 1)
                 ):
        """Generates levels using a tiling approach, i.e. moving through all of the cells, giving the network the surrounding ones and taking the prediction as the current tile.
        Args:
            game (Game): The game to generate for
            context_size (int, optional): How many tiles should the network take in as input to predict the current tile. 
                If this is 1, then the network takes in the surrounding 8 tiles (i.e. x +- 1 and y +- 1). 
                If this is two, then it takes in the surrounding 24 tiles.
                Defaults to 1.

            number_of_random_variables (int, optional): The number of random variables to add to the network. Defaults to 2.
            do_padding_randomly (bool, optional): If this is true, then we don't pad with -1s around the borders, but we instead make those random as well. Defaults to False.
            random_perturb_size (float, optional): If this is nonzero, all inputs to the net, including coordinates and surrounding tiles will be randomly perturbed by a gaussian (mean 0, variance 1) 
                multiplied by this value. Defaults to 0.
            predict_size (int, optional): If this is 1, then the network predicts one tile. If this is two, the network predicts 4 tiles (a 2x2 area), etc.
            reversed_direction (bool, optional): If this is 0, we iterate from the top left to the bottom right. Otherwise, if it is 1, we iterate from the bottom right to the top left.
            If it is 2, we choose random directions each time.
            use_one_hot_encoding (bool, optional). If this is true, then we use one hot encoding for the inputs instead of just ints. Defaults to False
            
            input_coords (bool, optional). If this is true, adds in two coordinates, between 0 and 1 to the input of the network
            add_bias (bool, optional). If true, then adds a bias term to the network.
            input_linear_coords (bool, optional). If true, adds in a normalised linear coordinate to the network, i.e. if you were to flatten the map into a 1D array, the fraction along it.
            
            normalisation_range_for_coordinates (Tuple[float, float], optional). This controls the normalisation range for the coordinates, i.e. what is the lowest and highest possible values. Defaults to (0, 1)

        """
        super().__init__(number_of_random_variables)
        self.game = game
        self.context_size = context_size
        self.do_padding_randomly = do_padding_randomly
        self.random_perturb_size = random_perturb_size
        self.tile_size = 2 * context_size + 1
        self.number_of_tile_types = len(self.game.level.tile_types)
        self.predict_size = predict_size
        self.reversed_direction = reversed_direction
        self.use_one_hot_encoding = use_one_hot_encoding
        self.input_coords = input_coords
        self.add_bias = add_bias

        self.input_linear_coords = input_linear_coords
        self.normalisation_range_for_coordinates = normalisation_range_for_coordinates

    def generate_level(self, net: LevelNeuralNet, input: np.ndarray, start_output: np.ndarray = None) -> Level:
        if not hasattr(self, 'input_linear_coords'):
            self.input_linear_coords = False
        
        h, w = self.game.level.height, self.game.level.width
        half_tile = self.tile_size // 2

        if self.do_padding_randomly:
            # Pad randomly, and don't make the edges special.
            output = np.random.randint(0, self.number_of_tile_types, size=(
                h + 2 * half_tile, w + 2 * half_tile))
        else:
            # pad it with negatives
            output = np.zeros((h + half_tile * 2, w + half_tile * 2)) - 1
            output[half_tile:-half_tile, half_tile:-
                   half_tile] = np.random.randint(0, self.number_of_tile_types, size=(h, w))
        
        if start_output is not None:
            output[half_tile:-half_tile, half_tile:-half_tile] = start_output
        else:
            pass
            # assert output[half_tile:-half_tile, half_tile:-half_tile].sum() != 0
        
        input_list = list(input)
        output[half_tile:-half_tile, half_tile:-half_tile] = 1 * \
            (output[half_tile:-half_tile, half_tile:-half_tile] > 0.5)
        X = self.predict_size - 1
        num_preds = self.predict_size ** 2

        range_rows = range(half_tile, h + half_tile - X)
        range_cols = range(half_tile, w + half_tile - X)
        if self.reversed_direction == 1:
            range_rows = reversed(range_rows)
            range_cols = reversed(range_cols)
        elif self.reversed_direction == 2:
            if np.random.rand() < 0.5:
                range_rows = reversed(range_rows)
            if np.random.rand() < 0.5:
                range_cols = reversed(range_cols)

        # This is super important, as the reversed thing is a one use iterator!
        # You cannot iterate multiple times!!!!!!!!!!
        range_rows = list(range_rows)
        range_cols = list(range_cols)

        for row in range_rows:
            for col in range_cols:
                # get state
                # Suppose (row, col) is the top left corner of our prediction tile. Then we need to move predict_size - 1 more to the right and down.

                little_slice = output[row - half_tile: row + half_tile +
                                      1 + X, col - half_tile: col + half_tile + 1 + X]
                # This should be a nxn slice now.
                assert little_slice.shape == (
                    self.tile_size + X, self.tile_size + X)
                total = self.tile_size * self.tile_size
                little_slice = little_slice.flatten()

                little_slice_list = list(little_slice)
                if self.predict_size == 1:  # Don't remove the middle tiles if pred size > 1
                    # Remove the middle element, which corresponds to the current cell.
                    little_slice_list.pop(total//2)
                    assert len(little_slice_list) == total - \
                        1, f"{len(little_slice)} != {total-1}"

                if self.use_one_hot_encoding:
                    # now we need to encode the array into a one hot thing.
                    curr_ans = []
                    for value in little_slice_list:
                        curr_ans = curr_ans + _one_hot(value, self.number_of_tile_types)
                    curr_ans = curr_ans
                    little_slice_list = curr_ans
                # Add in random input.
                little_slice_list.extend(input_list)

                input_to_net = little_slice_list

                if self.random_perturb_size != 0:
                    # Perturb input randomly.
                    input_to_net = np.add(input_to_net, np.random.randn(
                        len(input_to_net)) * self.random_perturb_size)
                
                if self.input_coords:
                    frac_y = self._normalise_coordinate((row - half_tile) / (h - 1))
                    frac_x = self._normalise_coordinate((col - half_tile) / (w - 1))
                    input_to_net = np.append(input_to_net, [frac_x, frac_y])

                if self.input_linear_coords:
                    _x, _y = (col - half_tile), (row - half_tile)
                    frac = (_y * self.game.level.width + _x) / (h * w - 1)
                    assert 0 <= frac <= 1, f"Bad frac {frac} {_x} {_y} {h} {w}"
                    frac = self._normalise_coordinate(frac)
                    input_to_net = np.append(input_to_net, [frac])

                if self.add_bias:
                    # add in the bias value of 1
                    input_to_net = np.append(input_to_net, [1])
                
                # Should be a result of size self.number_of_tile_types, so choose argmax.
                output_results = net.activate(input_to_net)
                assert len(output_results) == num_preds * \
                    self.number_of_tile_types, "Network should output the same amount of numbers as there are tile types"
                output_results = np.array(output_results).reshape(
                    (self.number_of_tile_types, self.predict_size, self.predict_size))

                tile_to_choose = np.argmax(output_results, axis=0)
                assert tile_to_choose.shape == (
                    self.predict_size, self.predict_size)
                for i in range(self.predict_size):
                    for j in range(self.predict_size):
                        output[row + i, col + j] = tile_to_choose[i, j]

        output = output[half_tile:-half_tile, half_tile:-half_tile]
        assert output.shape == (h, w)
        return self.game.level.from_map(output)

    def _normalise_coordinate(self, coord: float) -> float:
        assert 0 <= coord <= 1, f"Bad coord {coord}"
        # Normalises a coord (which is in the range from 0 to 1 to between self.normalisation_range_for_coordinates[0] and self.normalisation_range_for_coordinates[1])
        if not hasattr(self, 'normalisation_range_for_coordinates'):
            self.normalisation_range_for_coordinates = (0, 1)
        _norm_range = self.normalisation_range_for_coordinates[1] - self.normalisation_range_for_coordinates[0]
        return coord * _norm_range + self.normalisation_range_for_coordinates[0]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tile_size={self.tile_size}, number_of_random_variables={self.number_of_random_variables}, do_padding_randomly={self.do_padding_randomly}, random_perturb_size={self.random_perturb_size}, context_size={self.context_size}, number_of_tiles={self.number_of_tile_types}, game={self.game}, predict_size={self.predict_size}, reverse={self.reversed_direction}, use_one_hot_encoding={self.use_one_hot_encoding})"
    
    def get_input_output_size(self) -> Tuple[int, int]:
        return _get_inout_dim(self, DIM = 2)

class GenerateGeneralLevelUsingTiling3D(SeedableLevelGenerator):
    def __init__(self, game: Game,
                 context_size: int = 1, number_of_random_variables: int = 2,
                 do_padding_randomly: bool = False,
                 random_perturb_size: float = 0,
                 predict_size: int = 1,
                 reversed_direction: int = 0,
                 use_one_hot_encoding: int = False
                 ):
        """
        Does stuff in 3D
        Generates levels using a tiling approach, i.e. moving through all of the cells, giving the network the surrounding ones and taking the prediction as the current tile.
        Args:
            game (Game): The game to generate for
            context_size (int, optional): How many tiles should the network take in as input to predict the current tile. 
                If this is 1, then the network takes in the surrounding 8 tiles (i.e. x +- 1 and y +- 1). 
                If this is two, then it takes in the surrounding 24 tiles.
                Defaults to 1.

            number_of_random_variables (int, optional): The number of random variables to add to the network. Defaults to 2.
            do_padding_randomly (bool, optional): If this is true, then we don't pad with -1s around the borders, but we instead make those random as well. Defaults to False.
            random_perturb_size (float, optional): If this is nonzero, all inputs to the net, including coordinates and surrounding tiles will be randomly perturbed by a gaussian (mean 0, variance 1) 
                multiplied by this value. Defaults to 0.
            predict_size (int, optional): If this is 1, then the network predicts one tile. If this is two, the network predicts 4 tiles (a 2x2 area), etc.
            reversed_direction (bool, optional): If this is 0, we iterate from the top left to the bottom right. Otherwise, if it is 1, we iterate from the bottom right to the top left.
            If it is 2, we choose random directions each time.
            use_one_hot_encoding (bool, optional). If this is true, then we use one hot encoding for the inputs instead of just ints. Defaults to False
        """
        super().__init__(number_of_random_variables)
        self.game = game
        self.context_size = context_size
        self.do_padding_randomly = do_padding_randomly
        self.random_perturb_size = random_perturb_size
        self.tile_size = 2 * context_size + 1
        self.number_of_tile_types = len(self.game.level.tile_types)
        self.predict_size = predict_size
        self.reversed_direction = reversed_direction
        self.use_one_hot_encoding = use_one_hot_encoding

    def generate_level(self, net: LevelNeuralNet, input: np.ndarray, start_output: np.ndarray = None) -> Level:
        h, w, d = self.game.level.height, self.game.level.width, self.game.level.depth
        half_tile = self.tile_size // 2

        if self.do_padding_randomly:
            # Pad randomly, and don't make the edges special.
            output = np.random.randint(0, self.number_of_tile_types, size=(
                w + 2 * half_tile, h + 2 * half_tile, d + 2 * half_tile))
        else:
            # pad it with negatives
            output = np.zeros((w + half_tile * 2, h + half_tile * 2, d + half_tile * 2)) - 1
            output[half_tile:-half_tile, half_tile:-
                   half_tile, half_tile:-half_tile] = np.random.randint(0, self.number_of_tile_types, size=(w, h, d))

        if start_output is not None:
            output[half_tile:-half_tile, half_tile:-
                   half_tile, half_tile:-half_tile] = start_output
        else:
            pass
            # assert output[half_tile:-half_tile, half_tile:-half_tile].sum() != 0

        input_list = list(input)
        X = self.predict_size - 1
        num_preds = self.predict_size ** 3

        range_rows = range(half_tile, w + half_tile - X)
        range_cols = range(half_tile, h + half_tile - X)
        range_depth = range(half_tile, d + half_tile - X)
        if self.reversed_direction == 1:
            range_rows = reversed(range_rows)
            range_cols = reversed(range_cols)
            range_depth = reversed(range_depth)
        elif self.reversed_direction == 2:
            if np.random.rand() < 0.5:
                range_rows = reversed(range_rows)
            if np.random.rand() < 0.5:
                range_cols = reversed(range_cols)
            if np.random.rand() < 0.5:
                range_depth = reversed(range_depth)

        # This is super important, as the reversed thing is a one use iterator!
        # You cannot iterate multiple times!!!!!!!!!!
        range_rows = list(range_rows)
        range_cols = list(range_cols)
        range_depth = list(range_depth)

        for row in range_rows:
            for col in range_cols:
                for ddd in range_depth:
                    # get state
                    # Suppose (row, col) is the top left corner of our prediction tile. Then we need to move predict_size - 1 more to the right and down.

                    little_slice = output[row - half_tile: row + half_tile + 1 + X, 
                                          col - half_tile: col + half_tile + 1 + X,
                                          ddd - half_tile: ddd + half_tile + 1 + X]
                    # This should be a nxn slice now.
                    assert little_slice.shape == (
                        self.tile_size + X, self.tile_size + X, self.tile_size + X)
                    total = self.tile_size * self.tile_size * self.tile_size
                    little_slice = little_slice.flatten()

                    little_slice_list = list(little_slice)
                    if self.predict_size == 1:  # Don't remove the middle tiles if pred size > 1
                        # Remove the middle element, which corresponds to the current cell.
                        little_slice_list.pop(total//2)
                        assert len(little_slice_list) == total - \
                            1, f"{len(little_slice)} != {total-1}"

                    if self.use_one_hot_encoding:
                        # now we need to encode the array into a one hot thing.
                        curr_ans = []
                        for value in little_slice_list:
                            curr_ans = curr_ans + _one_hot(value, self.number_of_tile_types)
                        curr_ans = curr_ans
                        little_slice_list = curr_ans
                    # Add in random input.
                    little_slice_list.extend(input_list)

                    input_to_net = little_slice_list
                    # assert len(input_to_net) == total -1 + self.number_of_random_variables
                    if self.random_perturb_size != 0:
                        # Perturb input randomly.
                        input_to_net = np.add(input_to_net, np.random.randn(len(input_to_net)) * self.random_perturb_size)

                    # Should be a result of size self.number_of_tile_types, so choose argmax.
                    output_results = net.activate(input_to_net)
                    assert len(output_results) == num_preds * \
                        self.number_of_tile_types, "Network should output the same amount of numbers as there are tile types"
                    output_results = np.array(output_results).reshape(
                        (self.number_of_tile_types, self.predict_size, self.predict_size, self.predict_size))

                    tile_to_choose = np.argmax(output_results, axis=0)
                    assert tile_to_choose.shape == (self.predict_size, self.predict_size, self.predict_size)
                    for i in range(self.predict_size):
                        for j in range(self.predict_size):
                            for k in range(self.predict_size):
                                output[row + i, col + j, ddd + k] = tile_to_choose[i, j, k]

        output = output[half_tile:-half_tile, half_tile:-half_tile, half_tile:-half_tile]
        assert output.shape == (w, h, d)
        return self.game.level.from_map(output)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tile_size={self.tile_size}, number_of_random_variables={self.number_of_random_variables}, do_padding_randomly={self.do_padding_randomly}, random_perturb_size={self.random_perturb_size}, context_size={self.context_size}, number_of_tiles={self.number_of_tile_types}, game={self.game}, predict_size={self.predict_size}, reverse={self.reversed_direction}, use_one_hot_encoding={self.use_one_hot_encoding})"

    
    def __call__(self, net: LevelNeuralNet, input = None, start_output: np.ndarray = None) -> Level:
        if input is None: 
            input = np.random.randn(self.number_of_random_variables)
        return self.generate_level(net, input, start_output)
    
    def get_input_output_size(self) -> Tuple[int, int]:
        return _get_inout_dim(self, DIM = 3)

class GenerateLevelUsingTilingMultipleIterations(NeatLevelGenerator):
    def __init__(self, 
                 generator: SeedableLevelGenerator,
                 number_of_iterations: int = 50
                 ):
        """This generates a level using multiple iterations of the above methods.
        Args:
            generator (NeatLevelGenerator) The base generator to use
            number_of_iterations (int): How many iterations to go for.
        """
        super().__init__(generator.number_of_random_variables)
        self.game = generator.game
        self.generator = generator
        self.number_of_iterations = number_of_iterations


    def generate_level(self, net: LevelNeuralNet, input: np.ndarray) -> Level:
        base_level = self.generator.generate_level(net, input)
        for i in range(self.number_of_iterations - 1):
            base_level = self.generator.generate_level(net, input, start_output=base_level.map)
        return base_level

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(generator={self.generator}, number_of_iterations={self.number_of_iterations})"

    def get_input_output_size(self) -> Tuple[int, int]:
        return self.generator.get_input_output_size()