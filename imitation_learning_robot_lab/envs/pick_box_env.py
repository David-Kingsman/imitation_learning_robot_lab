import os
import time
from pathlib import Path
import numpy as np
import spatialmath as sm

import mujoco
import mujoco.viewer

from .env import Env

from ..arm.robot import Robot, UR5e
from ..arm.motion_planning import LinePositionParameter, OneAttitudeParameter, CartesianParameter, \
    QuinticVelocityParameter, TrajectoryParameter, TrajectoryPlanner
from ..utils import mj


class PickBoxEnv(Env):
    _name = "pick_box"
    _robot_type = "UR5e"
    _height = 480
    _width = 640
    _states = [
        "px",
        "py",
        "pz",
        "gripper"
    ]
    _cameras = [
        "top",
        "hand"
    ]
    _state_dim = 4
    _action_dim = 4

    def __init__(self, render_mode: str = "rgb_array"):
        super().__init__()

        self._sim_hz = 500
        self._control_hz = 25

        self._render_mode = render_mode

        self._latest_action = None
        self._render_cache = None

        scene_path = Path(__file__).parent.parent / Path("assets/scenes/pick_box_scene.xml")
        self._mj_model: mujoco.MjModel = mujoco.MjModel.from_xml_path(os.fspath(scene_path))
        self._mj_data: mujoco.MjData = mujoco.MjData(self._mj_model)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._robot: Robot = UR5e()
        self._robot_q = np.zeros(self._robot.dof)
        self._ur5e_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                                  "wrist_2_joint", "wrist_3_joint"]
        self._robot_T = sm.SE3()
        self._T0 = sm.SE3()

        self._mj_renderer: mujoco.Renderer = None
        self._mj_viewer: mujoco.viewer.Handle = None

        self._step_num = 0
        self._obj_t = np.zeros(3)

    def reset(self):

        mujoco.mj_resetData(self._mj_model, self._mj_data)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._robot.disable_base()
        self._robot.disable_tool()

        self._robot.set_base(mj.get_body_pose(self._mj_model, self._mj_data, "ur5e_base"))
        self._robot_q = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
        self._robot.set_joint(self._robot_q)
        [mj.set_joint_q(self._mj_model, self._mj_data, jn, self._robot_q[i]) for i, jn in
         enumerate(self._ur5e_joint_names)]
        mujoco.mj_forward(self._mj_model, self._mj_data)
        mj.attach(self._mj_model, self._mj_data, "attach", "2f85", self._robot.fkine(self._robot_q))
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._robot.set_tool(sm.SE3.Trans(0.0, 0.0, 0.15))
        self._robot_T = self._robot.fkine(self._robot_q)
        self._T0 = self._robot_T.copy()

        px_box = np.random.uniform(low=1.4, high=1.5)
        py_box = np.random.uniform(low=0.3, high=0.9)
        pz_box = 0.77
        T_Box = sm.SE3.Trans(px_box, py_box, pz_box)
        mj.set_free_joint_pose(self._mj_model, self._mj_data, "Box", T_Box)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        px_container = np.random.uniform(low=1.4, high=1.5)
        py_container = np.random.uniform(low=0.3, high=0.9)
        pz_container = 0.77
        while np.linalg.norm(
                np.array([px_box, py_box, pz_box] - np.array([px_container, py_container, pz_container]))) < 0.2:
            px_container = np.random.uniform(low=1.4, high=1.5)
            py_container = np.random.uniform(low=0.3, high=0.9)
            pz_container = 0.77
        T_container = sm.SE3.Trans(px_container, py_container, pz_container)

        container_eq_data = np.zeros(11)
        container_eq_data[3:6] = T_container.t
        container_eq_data[6:10] = T_container.UnitQuaternion()
        container_eq_data[-1] = 1.0
        mj.attach(self._mj_model, self._mj_data, "container_attach",
                  "container_free_joint", T_container, eq_data=container_eq_data)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._mj_renderer = mujoco.renderer.Renderer(self._mj_model, height=self._height, width=self._width)
        if self._render_mode == "human":
            self._mj_viewer = mujoco.viewer.launch_passive(self._mj_model, self._mj_data,
                                                           show_left_ui=False, show_right_ui=False)

        self._step_num = 0
        observation = self._get_observation()
        info = {"is_success": False}
        return observation, info

    def step(self, action):
        n_steps = self._sim_hz // self._control_hz
        if action is not None:
            self._latest_action = action

            Ti = self._T0 * sm.SE3.Trans(action[0], action[1], action[2])
            self._robot.move_cartesian(Ti)
            joint_position = self._robot.get_joint()
            self._mj_data.ctrl[:6] = joint_position
            action[3] = np.clip(action[3], 0, 1)
            self._mj_data.ctrl[6] = action[3] * 255.0
        mujoco.mj_step(self._mj_model, self._mj_data, n_steps)

        observation = self._get_observation()
        reward = 0.0
        terminated = False

        self._step_num += 1

        truncated = False
        if self._step_num > 10000:
            truncated = True

        info = {"is_success": terminated}
        return observation, reward, terminated, truncated, info

    def render(self):
        if self._render_mode == "human":
            self._mj_viewer.sync()

    def close(self):
        if self._mj_viewer is not None:
            self._mj_viewer.close()
        if self._mj_renderer is not None:
            self._mj_renderer.close()

    def seed(self, seed=None):
        pass

    def _get_observation(self):
        mujoco.mj_forward(self._mj_model, self._mj_data)

        for i in range(len(self._ur5e_joint_names)):
            self._robot_q[i] = mj.get_joint_q(self._mj_model, self._mj_data, self._ur5e_joint_names[i])[0]
        self._robot_T = self._T0.inv() * self._robot.fkine(self._robot_q)
        agent_pos = np.zeros(4, dtype=np.float32)
        agent_pos[:3] = self._robot_T.t
        agent_pos[3] = np.linalg.norm(self._mj_data.site('left_pad').xpos - self._mj_data.site('right_pad').xpos)

        self._mj_renderer.update_scene(self._mj_data, 0)
        image_top = self._mj_renderer.render()
        self._mj_renderer.update_scene(self._mj_data, 1)
        image_hand = self._mj_renderer.render()

        obs = {
            'pixels': {
                'top': image_top,
                'hand': image_hand
            },
            'agent_pos': agent_pos
        }
        self._render_cache = image_top
        return obs
