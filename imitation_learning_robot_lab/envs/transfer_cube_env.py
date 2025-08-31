import os
import time
from pathlib import Path
import numpy as np
import spatialmath as sm

import mujoco
import mujoco.viewer

from .env import Env

from ..arm.robot import Robot, VX300S, RobotWrapper
from ..arm.motion_planning import LinePositionParameter, OneAttitudeParameter, \
    CartesianParameter, \
    QuinticVelocityParameter, TrajectoryParameter, TrajectoryPlanner
from ..utils import mj


class TransferCubeEnv(Env):
    _name = "transfer_cube"
    _robot_type = "ALOHA"
    _height = 240
    _width = 320
    _states = [
        "left_px", "left_py", "left_pz", "left_rx", "left_ry", "left_rz", "left_gripper",
        "right_px", "right_py", "right_pz", "right_rx", "right_ry", "right_rz", "right_gripper"
    ]
    _cameras = [
        "overhead_cam",
        "wrist_cam_left",
        "wrist_cam_right"
    ]

    def __init__(self, render_mode: str = "rgb_array"):
        super().__init__()

        self._sim_hz = 500

        self._render_mode = render_mode

        scene_path = Path(__file__).parent.parent / Path("assets/scenes/transfer_cube_scene.xml")
        self._mj_model: mujoco.MjModel = mujoco.MjModel.from_xml_path(os.fspath(scene_path))
        self._mj_data: mujoco.MjData = mujoco.MjData(self._mj_model)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._left_robot = RobotWrapper()
        self._left_robot_q = np.zeros(self._left_robot.dof)
        self._left_robot_joint_names = ["left/waist", "left/shoulder", "left/elbow", "left/forearm_roll",
                                        "left/wrist_angle", "left/wrist_rotate"]
        self._left_robot_T = sm.SE3()
        self._left_T0 = sm.SE3()
        self._left_tool_joint_name = "left/left_finger"

        self._right_robot = RobotWrapper()
        self._right_robot_q = np.zeros(self._right_robot.dof)
        self._right_robot_joint_names = ["right/waist", "right/shoulder", "right/elbow", "right/forearm_roll",
                                         "right/wrist_angle", "right/wrist_rotate"]
        self._right_robot_T = sm.SE3()
        self._right_T0 = sm.SE3()
        self._right_tool_joint_name = "right/right_finger"

        self._mj_renderer: mujoco.Renderer = None
        self._mj_viewer: mujoco.viewer.Handle = None

        self._step_num = 0
        self._obj_t = np.zeros(3)

        self._ready_time = 0.4

    def reset(self):
        mujoco.mj_resetData(self._mj_model, self._mj_data)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._left_robot.disable_base()
        self._left_robot.disable_tool()

        self._right_robot.disable_base()
        self._right_robot.disable_tool()

        self._left_robot.set_base(mj.get_body_pose(self._mj_model, self._mj_data, "left/base_link"))
        self._left_robot_q = np.array([0.0, -0.5, 0, 0.0, 0.5, 0.0])
        self._left_robot.set_joint(self._left_robot_q)
        [mj.set_joint_q(self._mj_model, self._mj_data, jn, self._left_robot_q[i]) for i, jn in
         enumerate(self._left_robot_joint_names)]
        mj.set_joint_q(self._mj_model, self._mj_data, "left/left_finger", 0.06)
        mj.set_joint_q(self._mj_model, self._mj_data, "left/right_finger", 0.06)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._left_robot.set_tool(sm.SE3.Trans([0.13, 0.0, -0.003]))
        self._left_robot_T = self._left_robot.fkine(self._left_robot_q)
        self._left_T0 = self._left_robot_T.copy()

        self._right_robot.set_base(mj.get_body_pose(self._mj_model, self._mj_data, "right/base_link"))
        self._right_robot_q = np.array([0.0, -0.5, 0.0, 0.0, 0.5, 0.0])
        self._right_robot.set_joint(self._right_robot_q)
        [mj.set_joint_q(self._mj_model, self._mj_data, jn, self._right_robot_q[i]) for i, jn in
         enumerate(self._right_robot_joint_names)]
        mj.set_joint_q(self._mj_model, self._mj_data, "right/left_finger", 0.06)
        mj.set_joint_q(self._mj_model, self._mj_data, "right/right_finger", 0.06)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._right_robot.set_tool(sm.SE3.Trans([0.13, 0.0, -0.003]))
        self._right_robot_T = self._right_robot.fkine(self._right_robot_q)
        self._right_T0 = self._right_robot_T.copy()

        mj_ctrl = np.zeros(14)
        mj_ctrl[:6] = self._left_robot_q
        mj_ctrl[6] = 0.06
        mj_ctrl[7:13] = self._right_robot_q
        mj_ctrl[13] = 0.06
        mujoco.mj_setState(self._mj_model, self._mj_data, mj_ctrl, mujoco.mjtState.mjSTATE_CTRL)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        px_cube = np.random.uniform(low=-0.2, high=0.0)
        py_cube = np.random.uniform(low=-0.3, high=0.1)
        pz_cube = 0.03
        T_cube = sm.SE3.Trans(px_cube, py_cube, pz_cube)
        mj.set_free_joint_pose(self._mj_model, self._mj_data, "cube_free_joint", T_cube)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        px_container = np.random.uniform(low=0.0, high=0.2)
        py_container = np.random.uniform(low=py_cube + 0.1, high=0.3)

        pz_container = 0.0
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

        while self._mj_data.time < self._ready_time:
            self.step(None)

        info = {"is_success": False}
        return observation, info

    def step(self, action):
        n_steps = self._sim_hz // self._control_hz

        # for i in range(n_steps):
        if action is not None:
            left_Ti = self._left_T0 * sm.SE3.Rt(R=sm.SO3.RPY(action[3], action[4], action[5]), t=action[:3])
            self._left_robot.move_cartesian(left_Ti)
            left_joint_position = self._left_robot.get_joint()
            self._mj_data.ctrl[:6] = left_joint_position
            action[6] = np.clip(action[6], 0, 1)
            self._mj_data.ctrl[6] = action[6] * (0.002 - 0.06) + 0.06

            right_Ti = self._right_T0 * sm.SE3.Rt(R=sm.SO3.RPY(action[10], action[11], action[12]), t=action[7:10])
            self._right_robot.move_cartesian(right_Ti)
            right_joint_position = self._right_robot.get_joint()
            self._mj_data.ctrl[7:13] = right_joint_position
            action[13] = np.clip(action[13], 0, 1)
            self._mj_data.ctrl[13] = action[13] * (0.002 - 0.06) + 0.06

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

        agent_pos = np.zeros(14, dtype=np.float32)

        for i in range(len(self._left_robot_joint_names)):
            self._left_robot_q[i] = mj.get_joint_q(self._mj_model, self._mj_data, self._left_robot_joint_names[i])[0]
        left_robot_T = self._left_T0.inv() * self._left_robot.fkine(self._left_robot_q)
        agent_pos[:3] = left_robot_T.t
        agent_pos[3: 6] = left_robot_T.rpy()
        agent_pos[6] = mj.get_joint_q(self._mj_model, self._mj_data, self._left_tool_joint_name)[0]

        for i in range(len(self._right_robot_joint_names)):
            self._right_robot_q[i] = mj.get_joint_q(self._mj_model, self._mj_data, self._right_robot_joint_names[i])[0]
        right_robot_T = self._right_T0.inv() * self._right_robot.fkine(self._right_robot_q)
        agent_pos[7: 10] = right_robot_T.t
        agent_pos[10: 13] = right_robot_T.rpy()
        agent_pos[13] = mj.get_joint_q(self._mj_model, self._mj_data, self._right_tool_joint_name)[0]

        overhead_cam_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam")
        worms_eye_cam_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "worms_eye_cam")
        teleoperator_pov_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "teleoperator_pov")
        collaborator_pov_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "collaborator_pov")
        wrist_cam_left_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam_left")
        wrist_cam_right_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam_right")

        self._mj_renderer.update_scene(self._mj_data, overhead_cam_id)
        image_overhead_cam = self._mj_renderer.render()
        # self._mj_renderer.update_scene(self._mj_data, worms_eye_cam_id)
        # image_worms_eye_cam = self._mj_renderer.render()
        # self._mj_renderer.update_scene(self._mj_data, teleoperator_pov_id)
        # image_teleoperator_pov = self._mj_renderer.render()
        # self._mj_renderer.update_scene(self._mj_data, collaborator_pov_id)
        # image_collaborator_pov = self._mj_renderer.render()
        self._mj_renderer.update_scene(self._mj_data, wrist_cam_left_id)
        image_wrist_cam_left = self._mj_renderer.render()
        self._mj_renderer.update_scene(self._mj_data, wrist_cam_right_id)
        image_wrist_cam_right = self._mj_renderer.render()

        obs = {
            'pixels': {
                'overhead_cam': image_overhead_cam,
                # 'worms_eye_cam': image_worms_eye_cam,
                # 'teleoperator_pov': image_teleoperator_pov,
                # 'collaborator_pov': image_collaborator_pov,
                'wrist_cam_left': image_wrist_cam_left,
                'wrist_cam_right': image_wrist_cam_right
            },
            'agent_pos': agent_pos
        }
        return obs

    def run(self):
        pass


if __name__ == '__main__':
    pass
