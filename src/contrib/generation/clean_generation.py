import itertools
from typing import Any, Dict, List, Tuple, TypedDict
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
    
    if hasattr(self, 'input_center_tile') and self.input_center_tile:
        in_dim = (t_s**DIM) * one_hot_factor + self.number_of_random_variables
    
        
    if hasattr(self, 'have_progress_indicator') and self.have_progress_indicator:
        assert hasattr(self, 'input_center_tile') and self.input_center_tile
        in_dim = (t_s**DIM) * (one_hot_factor + 1) + self.number_of_random_variables
    
    if self.predict_size > 1:
        in_dim = one_hot_factor * (t_s + self.predict_size - 1)**DIM + self.number_of_random_variables
    out_dim = p_s * len(self.game.level.tile_types)
    if hasattr(self, 'input_coords') and self.input_coords:
        in_dim += DIM
    if hasattr(self, 'input_linear_coords') and self.input_linear_coords:
        in_dim += 1
    if hasattr(self, 'add_bias') and self.add_bias:
        in_dim += 1
        
    if hasattr(self, 'use_memory') and self.use_memory and hasattr(self, 'mem_size'):
        in_dim += self.mem_size
        out_dim += self.mem_size
    
    
    if hasattr(self, 'use_hidden_channels') and self.use_hidden_channels and hasattr(self, 'hidden_channel_dim'):
        in_dim += (t_s**DIM) * self.hidden_channel_dim
        out_dim += p_s * self.hidden_channel_dim
    
    
    return in_dim, out_dim
        

def _one_hot(value: int, total: int) -> List[float]:
    arr = [0] * total
    arr[int(value)] = 1.0
    return arr

class MemoryParams(TypedDict):
    use_memory: bool
    mem_dim: int

class HiddenChannelsParams(TypedDict):
    use_hidden_channels: bool
    hidden_channel_dim: int
    hidden_channels_random_boundary: bool

class GenerateTilingLevelBase(SeedableLevelGenerator):
    def __init__(self, 
                 DIM: int,
                 game: Game,
                 context_size: int = 1, number_of_random_variables: int = 2,
                 do_padding_randomly: bool = False,
                 random_perturb_size: float = 0,
                 predict_size: int = 1,
                 reversed_direction: int = 0,
                 use_one_hot_encoding: int = False,
                 input_coords: bool = False,
                 add_bias: bool = False,
                 input_linear_coords: bool = False,
                 normalisation_range_for_coordinates: Tuple[float, float] = (0, 1),
                 memory_kwargs: MemoryParams = {},
                 hidden_channel_kwargs: HiddenChannelsParams = {},
                 double_buffered: bool = False,
                 input_center_tile: bool = False,
                 
                 have_progress_indicator: bool = False,
                 progress_how_many_steps_needed: int = 1,
                 progress_default_tile: int = 1,
                 start_level_with_default_tile: bool = False
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
            
            
            memory_kwargs: MemoryParams = {}. Parameters to control memory
            hidden_channel_kwargs: HiddenChannelsParams = {}. Parameters to control the hidden channels
            double_buffered (bool): If true, then the generated tiles are not put into the same buffer the tiles are being read from, there is a separate read and write one.
            input_center_tile (bool): If true, the network receives the center tile as well as the surrounding ones.
            
            have_progress_indicator (bool): If True, then the tiles are given as [tile_num, progress], where progress goes from 0 to 1.
            progress_how_many_steps_needed (int):  If `have_progress_indicator` is True, then this indicates how many times the network should use the same action on a specific tile for it to work.

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
        self.DIM = DIM
        
        self.memory_kwargs = memory_kwargs
        self.use_memory = self.memory_kwargs.get('use_memory', False)
        self.mem_size = self.memory_kwargs.get('mem_dim', 2)
        print("Using mem dim of ", self.mem_size, 'with memory enabled=', self.use_memory)
        self.mem_vector = np.zeros(self.mem_size, dtype=np.float32)
        
        self.hidden_channel_kwargs = hidden_channel_kwargs
        self.use_hidden_channels = self.hidden_channel_kwargs.get('use_hidden_channels', False)
        self.hidden_channel_dim = self.hidden_channel_kwargs.get('hidden_channel_dim', 2)
        self.prior_hidden_channels = None
        self.double_buffered = double_buffered
        self.input_center_tile = input_center_tile
        
        self.have_progress_indicator = have_progress_indicator
        if type(progress_how_many_steps_needed) == int:
            progress_how_many_steps_needed = {t: progress_how_many_steps_needed for t in self.game.level.tile_types}
            
        self.progress_how_many_steps_needed = progress_how_many_steps_needed
        self.progress_default_tile = progress_default_tile
        self.start_level_with_default_tile = start_level_with_default_tile
        self.progress_grid = None

    def _get_level_dimensions(self) -> List[int]:
        return list(self.game.level.map.shape)
    
    def generate_level(self, net: LevelNeuralNet, input: np.ndarray, start_output: np.ndarray = None) -> Level:
        use_memory          = hasattr(self, 'use_memory') and self.use_memory
        use_hidden_channels = hasattr(self, 'use_hidden_channels') and self.use_hidden_channels
        double_buffered     = hasattr(self, 'double_buffered') and self.double_buffered
        input_center_tile   = hasattr(self, 'input_center_tile') and self.input_center_tile
        do_progress         = hasattr(self, 'have_progress_indicator') and self.have_progress_indicator
        start_level_with_default_tile         = hasattr(self, 'start_level_with_default_tile') and self.start_level_with_default_tile
        
        
        
        if use_memory and start_output is None: self.mem_vector = np.zeros(self.mem_size, dtype=np.float32)
    
        if not hasattr(self, 'input_linear_coords'):
            self.input_linear_coords = False
        half_tile = self.tile_size // 2
        
        # h, w = self.game.level.height, self.game.level.width
        level_dimensions = self._get_level_dimensions()
        shape_to_generate = tuple(a + 2 * half_tile for a in level_dimensions)
        actual_level_slice = tuple([slice(half_tile, -half_tile, None) for _ in range(self.DIM)])
        
        if self.do_padding_randomly:
            # Pad randomly, and don't make the edges special.
            output = np.random.randint(0, self.number_of_tile_types, size=shape_to_generate)
        else:
            # pad it with negatives
            output = np.zeros(shape_to_generate) - 1
            output[actual_level_slice] = np.random.randint(0, self.number_of_tile_types, size=level_dimensions)
        
        if start_output is not None:
            output[actual_level_slice] = start_output
        elif start_level_with_default_tile:
            output[actual_level_slice] = np.ones_like(output[actual_level_slice]) * self.progress_default_tile
        
        if use_hidden_channels and start_output is None:
            
            # Here we must recreate the self.prior_hidden_channels
            _shape = output.shape
            _new_shape = _shape + (self.hidden_channel_dim, )
            self.prior_hidden_channels = np.zeros(_new_shape, dtype=np.float32)
            
            if self.hidden_channel_kwargs.get('hidden_channels_random_boundary', False):
                
                new = np.random.randn(*_new_shape)
                _s = actual_level_slice + (slice(None, None, None), )
                new[_s] = self.prior_hidden_channels[_s]
                self.prior_hidden_channels = new
                
            
        if double_buffered:
            output_write_buffer     = np.zeros_like(output)
            hidden_channels_write_buffer  = np.zeros_like(self.prior_hidden_channels)
        
        if do_progress: 
            assert input_center_tile == True
            assert double_buffered == False
            assert self.predict_size == 1
        
        if do_progress and start_output is None:
            _shape = output.shape
            _new_shape = _shape + (2, )
            self.progress_grid = np.ones(_new_shape, dtype=np.float32)
            slice_to_use = tuple(slice(None, None, None) for _ in range(len(output.shape))) + (0, )
            if self.start_level_with_default_tile:
                self.progress_grid[slice_to_use] = np.ones_like(output) * self.progress_default_tile
            else:
                self.progress_grid[slice_to_use] = output
        
        if do_progress:
            output = self.progress_grid
        
        input_list = list(input)
        X = self.predict_size - 1
        num_preds = self.predict_size ** self.DIM

        ranges_to_iterate_over = [
            range(half_tile, a + half_tile - X) for a in level_dimensions
        ]
        
        if self.reversed_direction == 1:
            ranges_to_iterate_over = [reversed(r) for r in ranges_to_iterate_over]
        elif self.reversed_direction == 2:
            for i in range(self.DIM):
                
                if np.random.rand() < 0.5:
                    ranges_to_iterate_over[i] = reversed(ranges_to_iterate_over[i])

        # This is super important, as the reversed thing is a one use iterator!
        # You cannot iterate multiple times!!!!!!!!!!
        ranges_to_iterate_over = [list(r) for r in ranges_to_iterate_over]
        all_coords_to_iterate_over = itertools.product(*ranges_to_iterate_over)
        CORRECT_SHAPE = tuple(self.tile_size + X for _ in range(self.DIM))
        correct_pred_size = [self.predict_size] * self.DIM
        OUTPUT_SHAPE = tuple([self.number_of_tile_types] + correct_pred_size)
        
        if do_progress:
            CORRECT_SHAPE = CORRECT_SHAPE + (2, )
        
        extra_hidden_outputs = 0 if not use_hidden_channels else self.hidden_channel_dim * num_preds
        for current_tile_coord in all_coords_to_iterate_over:
            # get state
            # Suppose (row, col) is the top left corner of our prediction tile. Then we need to move predict_size - 1 more to the right and down.

            curr_slice = tuple([slice(a - half_tile, a + half_tile + 1 + X, None) for a in current_tile_coord])

            little_slice = output[curr_slice]
            # This should be a nxn slice now.
            assert little_slice.shape == CORRECT_SHAPE
            total = self.tile_size ** self.DIM
            little_slice_not_flat = little_slice
            little_slice = little_slice.ravel()

            little_slice_list = (little_slice).tolist()
            if self.predict_size == 1:  # Don't remove the middle tiles if pred size > 1
                # Remove the middle element, which corresponds to the current cell.
                if input_center_tile:
                    pass # do not remove
                else:
                    # Remove the center tile
                    little_slice_list.pop(total//2)
                    assert len(little_slice_list) == total - \
                        1, f"{len(little_slice)} != {total-1}"

            if self.use_one_hot_encoding:
                # now we need to encode the array into a one hot thing.
                curr_ans = []
                if do_progress:
                    assert input_center_tile
                    temp_to_use_now = little_slice_not_flat.reshape(-1, 2)
                    for value in temp_to_use_now:
                        assert len(value) == 2
                        curr_ans = curr_ans + _one_hot(value[0], self.number_of_tile_types) + list(value[1:])
                else:
                    for value in little_slice_list:
                        # Default
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
                coords = [self._normalise_coordinate((coord - half_tile) / max(1, size - 1)) for coord, size in zip(current_tile_coord, level_dimensions)]
                input_to_net = np.append(input_to_net, coords)

            if self.input_linear_coords:
                temp = [a - half_tile for a in current_tile_coord]
                max_linear_index = np.prod(level_dimensions) - 1
                
                linear_index = np.ravel_multi_index(temp, level_dimensions)
                
                frac = (linear_index) / (max_linear_index)
                assert 0 <= frac <= 1, f"Bad frac {frac} {frac} {linear_index} {max_linear_index} {temp}"
                frac = self._normalise_coordinate(frac)
                input_to_net = np.append(input_to_net, [frac])

            if self.add_bias:
                # add in the bias value of 1
                input_to_net = np.append(input_to_net, [1])
            
            if use_memory:
                input_to_net = np.append(input_to_net, self.mem_vector)
            
            if use_hidden_channels:
                tmp_vec = self.prior_hidden_channels[curr_slice]
                assert tmp_vec.shape == CORRECT_SHAPE + (self.hidden_channel_dim, )
                input_to_net = np.append(input_to_net, tmp_vec.ravel())
                
            # Should be a result of size self.number_of_tile_types, so choose argmax.
            output_results = net.activate(input_to_net)
            
            desired_tiles = num_preds * self.number_of_tile_types + (0 if not (use_memory) else self.mem_size) + extra_hidden_outputs
            if not (use_memory or use_hidden_channels): assert desired_tiles == num_preds * self.number_of_tile_types # sanity check
            
            assert len(output_results) == desired_tiles, "Network should output the same amount of numbers as there are tile types"
            
            if use_memory:
                self.mem_vector = output_results[-self.mem_size:]
                output_results = output_results[:-self.mem_size]
                assert len(self.mem_vector) == self.mem_size
            
            tile_slice_to_place_now = tuple([slice(a, a+self.predict_size, None) for a in current_tile_coord])
            
            if use_hidden_channels:
                new_output_channels = output_results[-extra_hidden_outputs:]
                assert len(new_output_channels) == self.hidden_channel_dim
                output_results = output_results[:-extra_hidden_outputs]
                # Put these new hidden channels into the array.
                if double_buffered:
                    hidden_channels_write_buffer[tile_slice_to_place_now] = np.array(new_output_channels).reshape(-1, self.hidden_channel_dim)
                else:
                    self.prior_hidden_channels[tile_slice_to_place_now] = np.array(new_output_channels).reshape(-1, self.hidden_channel_dim)
            
            output_results = np.array(output_results).reshape(OUTPUT_SHAPE)

            tile_to_choose = output_results.argmax(axis=0)
            assert tile_to_choose.shape == tuple(correct_pred_size)
            
            if do_progress:
                tile_already_placed, progress = output[tile_slice_to_place_now].squeeze()
                new_tile = tile_to_choose.squeeze().item()
                new_progress = None
                delta_prog = 1 / self.progress_how_many_steps_needed[new_tile]
                if tile_already_placed == tile_to_choose:
                    new_progress = np.clip(progress + delta_prog, 0, 1)
                else:
                    new_progress = delta_prog
                
                output[tile_slice_to_place_now + (0,)] = new_tile
                output[tile_slice_to_place_now + (1,)] = new_progress
            else:
                if double_buffered:
                    output_write_buffer[tile_slice_to_place_now] = tile_to_choose
                else:
                    output[tile_slice_to_place_now] = tile_to_choose

        if double_buffered:
            # swap the buffers
            output                      = output_write_buffer
            self.prior_hidden_channels  = hidden_channels_write_buffer
        
        if do_progress: 
            # Need this copy, because we modify output later.
            self.progress_grid = output.copy()
            _tmp = tuple(slice(None, None, None) for _ in range(len(output.shape) - 1)) 
            _tile = _tmp + (0,)
            _prog = _tmp + (1,)

            output = output[_tile]
            output[self.progress_grid[_prog] < 1.0] = self.progress_default_tile
        
        output = output[actual_level_slice]
        assert output.shape == tuple(level_dimensions)
        return self.game.level.from_map(output)

    def _normalise_coordinate(self, coord: float) -> float:
        assert 0 <= coord <= 1, f"Bad coord {coord}"
        # Normalises a coord (which is in the range from 0 to 1 to between self.normalisation_range_for_coordinates[0] and self.normalisation_range_for_coordinates[1])
        if not hasattr(self, 'normalisation_range_for_coordinates'):
            self.normalisation_range_for_coordinates = (0, 1)
        _norm_range = self.normalisation_range_for_coordinates[1] - self.normalisation_range_for_coordinates[0]
        return coord * _norm_range + self.normalisation_range_for_coordinates[0]

    def to_dict(self):
        return dict(
                DIM=self.DIM,
                game=self.game,
                context_size=self.context_size,
                do_padding_randomly=self.do_padding_randomly,
                random_perturb_size=self.random_perturb_size,
                tile_size=self.tile_size,
                number_of_tile_types=self.number_of_tile_types,
                predict_size=self.predict_size,
                reversed_direction=self.reversed_direction,
                use_one_hot_encoding=self.use_one_hot_encoding,
                input_coords=self.input_coords,
                add_bias=self.add_bias,
                input_linear_coords=self.input_linear_coords,
                normalisation_range_for_coordinates=self.normalisation_range_for_coordinates,
        )
    
    def __repr__(self) -> str:
        d = self.to_dict()
        s = ', '.join(f'{key}={val}' for key, val in d.items())
        return f"{self.__class__.__name__}({s})"
    
    def get_input_output_size(self) -> Tuple[int, int]:
        return _get_inout_dim(self, DIM = self.DIM)

class GenerateTilingLevel2D(GenerateTilingLevelBase): 
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)
        
class GenerateTilingLevel3D(GenerateTilingLevelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)
