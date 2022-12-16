from typing import Dict, List, Tuple
import numpy as np
from games.level import Level
from novelty_neat.fitness.fitness import IndependentNeatFitnessFunction
import skimage.morphology as morph
from contrib.novelty.extra_distance_functions import get_tiles_from_levels


class HasPatternsFitness(IndependentNeatFitnessFunction):
    """
    This is a fitness that counts pattern occurrences.
    """
    def __init__(self, *args, patterns: List[Tuple[np.ndarray, float]], **kwargs):
        """_summary_

        Args:
            patterns (List[Tuple[np.ndarray, float]]): A list of (pattern, prob) tuples, indicating how often each pattern is desired. The pattern is just a 2D numpy array.
        """
        super().__init__(*args, **kwargs)
        self.patterns = patterns
    
    def calc_fitness_single_level(self, level: Level) -> float:
        if len(level.map.shape) == 3:
            assert level.map.shape[1] == 1
            map = level.map[:, 0, :]
        else:
            map = level.map
        
        all_patterns: Dict[Tuple[int, int], List[np.ndarray]] = {}
        counts = np.zeros(len(self.patterns), dtype=np.float32)
        desired_probs = np.zeros(len(self.patterns), dtype=np.float32)
        for i, (desired_pat, desired_prob) in enumerate(self.patterns):
            size = desired_pat.shape
            if size not in all_patterns:
                all_patterns[size] = get_tiles_from_levels(map, size)
            
            patterns_from_this_level = all_patterns[size]
            
            for curr_pat in patterns_from_this_level:
                counts[i] += np.all(desired_pat == curr_pat)
            desired_probs[i] = desired_prob
        
        counts /= len(patterns_from_this_level)
        counts /= max(counts.sum(), 1)
        diff = (np.abs(desired_prob - counts)).sum()
        return 1 - np.clip(diff, 0, 1)
