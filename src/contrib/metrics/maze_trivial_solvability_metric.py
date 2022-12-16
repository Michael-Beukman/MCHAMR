from typing import List
from games.game import Game
from games.level import Level
from metrics.metric import Metric


class MazeTrivialSolvability(Metric):
    """1 if trivial maze path, else 0
    """

    def __init__(self, game: Game) -> None:
        super().__init__(game)

    def evaluate(self, levels: List[Level]) -> List[float]:
        ans = []
        for l in levels:
            filled = l.tile_types_reversed['filled']
            empty = l.tile_types_reversed['empty']
            a = (l.map[0, :] == empty).sum() + (l.map[:, -1] == empty).sum()
            b = (l.map[-1, :] == empty).sum() + (l.map[:, 0] == empty).sum()
            
            max_a = (l.map[0, :]).size + (l.map[:, -1]).size
            max_b = (l.map[-1, :]).size + (l.map[:, 0]).size
            if a == max_a or b == max_b:
                ans.append(1)
            else:
                ans.append(0)
        return ans

    def name(self):
        return "MazeTrivialSolvability"
