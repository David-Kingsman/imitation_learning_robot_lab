from enum import unique
from imitation_learning_robot_lab.arm.interface import ModeEnum


@unique
class PathPlanningModeEnum(ModeEnum):
    JOINT = 'joint'
    CARTESIAN = 'cartesian'
