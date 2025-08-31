from enum import unique
from imitation_learning_lerobot.arm.interface import ModeEnum


@unique
class VelocityPlanningModeEnum(ModeEnum):
    CUBIC = 'cubic'
    QUINTIC = 'quintic'
    T_CURVE = 't_curve'
