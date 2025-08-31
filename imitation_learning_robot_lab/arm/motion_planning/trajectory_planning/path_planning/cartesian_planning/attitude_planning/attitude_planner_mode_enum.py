from enum import unique
from imitation_learning_lerobot.arm.interface import ModeEnum


@unique
class AttitudePlannerModeEnum(ModeEnum):
    ONE = 'one'
    TWO = 'two'
    THREE = 'three'
