from abc import ABC

from imitation_learning_robot_lab.arm.interface import Parameter


class PositionParameter(Parameter, ABC):
    def get_length(self):
        pass
