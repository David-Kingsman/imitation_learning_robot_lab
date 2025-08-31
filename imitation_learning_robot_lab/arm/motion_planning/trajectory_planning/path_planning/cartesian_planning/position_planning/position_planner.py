import numpy as np

from imitation_learning_robot_lab.arm.interface import StrategyWrapper


class PositionPlanner(StrategyWrapper):

    def interpolate(self, s) -> np.ndarray:
        return self.strategy.interpolate(s)
