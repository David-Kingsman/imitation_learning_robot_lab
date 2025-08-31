from abc import ABC

from imitation_learning_lerobot.arm.interface import Parameter


class PathParameter(Parameter, ABC):
    def get_length(self):
        pass
