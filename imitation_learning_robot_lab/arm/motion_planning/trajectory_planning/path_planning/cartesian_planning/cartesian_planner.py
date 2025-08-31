from spatialmath import SE3

from imitation_learning_robot_lab.arm.interface import ModeEnum

from ..path_planner_strategy import PathPlannerStrategy

from .cartesian_parameter import CartesianParameter

from .position_planning import PositionPlanner
from .attitude_planning import AttitudePlanner

from ..path_planning_mode_enum import PathPlanningModeEnum


class CartesianPlanner(PathPlannerStrategy):

    def __init__(self, parameter: CartesianParameter):
        super().__init__(parameter)
        self.position_planner: PositionPlanner = PositionPlanner(parameter.get_position_parameter())
        self.attitude_planner: AttitudePlanner = AttitudePlanner(parameter.get_attitude_parameter())

    def interpolate(self, s) -> SE3:
        t = self.position_planner.interpolate(s)
        R = self.attitude_planner.interpolate(s)
        T: SE3 = SE3(*t) * SE3(R)

        return T

    @classmethod
    def mode(cls) -> ModeEnum:
        return PathPlanningModeEnum.CARTESIAN
