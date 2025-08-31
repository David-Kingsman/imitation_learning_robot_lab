from enum import unique
from imitation_learning_lerobot.arm.interface import ModeEnum


@unique
class PositionPlanningModeEnum(ModeEnum):
    LINE = 'line'
    ARC_CENTER = 'arc_center'
    ARC_POINT = 'arc_point'
