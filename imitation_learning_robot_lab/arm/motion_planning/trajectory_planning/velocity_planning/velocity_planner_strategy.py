from abc import ABC

from imitation_learning_robot_lab.arm.interface import Strategy, Parameter


class VelocityPlannerStrategy(Strategy, ABC):

    def __init__(self, parameter: Parameter):
        super().__init__(parameter)
