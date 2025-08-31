import os
from pathlib import Path

import numpy as np
import placo
from spatialmath import SE3

from .robot import Robot, get_transformation_mdh, wrap


class RobotWrapper(Robot):
    def __init__(self) -> None:
        super().__init__()

        self._dof = 6

        urdf_dir = Path(__file__).parent.parent / Path("assets/vx300s")
        self._robot_wrapper = placo.RobotWrapper(os.fspath(urdf_dir), placo.Flags.ignore_collisions)

        self._joint_names = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]

        self._solver = placo.KinematicsSolver(self._robot_wrapper)
        self._solver.mask_fbase(True)
        self._effector_task = self._solver.add_frame_task("gripper_link", np.eye(4))
        self._effector_task.configure("gripper_link", "soft", 1.0, 0.1)

        self._manipulability = self._solver.add_manipulability_task("gripper_link", "both", 1.0)
        self._manipulability.configure("manipulability", "soft", 1e-1)

        self._solver.enable_joint_limits(True)
        self._solver.enable_velocity_limits(True)
        self._solver.dt = 0.04

    def fkine(self, q) -> SE3:
        for i, joint_name in enumerate(self._joint_names):
            self._robot_wrapper.set_joint(joint_name, q[i])
        self._robot_wrapper.update_kinematics()
        return self._base * SE3(self._robot_wrapper.get_T_world_frame("gripper_link")) * self._tool

    def ikine(self, Twt: SE3) -> np.ndarray:
        self._effector_task.T_world_frame = (self._base.inv() * Twt * self._tool.inv()).A
        q = self.q0.copy()
        try:
            self._solver.solve(True)
            self._robot_wrapper.update_kinematics()
            for i, joint_name in enumerate(self._joint_names):
                q[i] = self._robot_wrapper.get_joint(joint_name)
        except Exception:
            pass
        return q

    def set_joint(self, q):
        super().set_joint(q)
        for i, joint_name in enumerate(self._joint_names):
            self._robot_wrapper.set_joint(joint_name, q[i])
        self._robot_wrapper.update_kinematics()

    def set_base(self, base: SE3):
        self._base = base.copy()

    def set_tool(self, tool: SE3):
        self._tool = tool.copy()

    def disable_base(self):
        self._base = SE3()

    def disable_tool(self):
        self._tool = SE3()
