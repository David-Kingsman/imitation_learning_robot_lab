from abc import ABC

from imitation_learning_robot_lab.arm.interface import Strategy

from .path_parameter import PathParameter


class PathPlannerStrategy(Strategy, ABC):

    def __init__(self, parameter: PathParameter):
        super().__init__(parameter)
