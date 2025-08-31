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


class PickAndPlaceEnv(Env):
    _name = "pick_and_place"
    _robot_type = "UR5e"
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

    def __init__(self, render_mode: str = "rgb_array"):
        super().__init__()

        self._sim_hz = 500
        self._control_hz = 25

        self._render_mode = render_mode

        self._latest_action = None
        self._render_cache = None

        scene_path = Path(__file__).parent.parent / Path("assets/scenes/scene.xml")
        self._mj_model: mujoco.MjModel = mujoco.MjModel.from_xml_path(os.fspath(scene_path))
        self._mj_data: mujoco.MjData = mujoco.MjData(self._mj_model)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._robot: Robot = UR5e()
        self._robot_q = np.zeros(self._robot.dof)
        self._ur5e_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                                  "wrist_2_joint", "wrist_3_joint"]
        self._robot_T = sm.SE3()
        self._T0 = sm.SE3()

        self._height = 480
        self._width = 640
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

        px = np.random.uniform(low=1.4, high=1.5)
        py = np.random.uniform(low=0.3, high=0.9)
        pz = 0.77
        T_Box = sm.SE3.Trans(px, py, pz)
        mj.set_free_joint_pose(self._mj_model, self._mj_data, "Box", T_Box)
        mujoco.mj_forward(self._mj_model, self._mj_data)
        self._obj_t = mj.get_body_pose(self._mj_model, self._mj_data, "Box").t

        self._mj_renderer = mujoco.renderer.Renderer(self._mj_model, height=self._height, width=self._width)
        if self._render_mode == "human":
            self._mj_viewer = mujoco.viewer.launch_passive(self._mj_model, self._mj_data)

        self._step_num = 0
        observation = self._get_observation()
        info = {"is_success": False}
        return observation, info

    def step(self, action):
        n_steps = self._sim_hz // self._control_hz
        if action is not None:
            self._latest_action = action
            for i in range(n_steps):
                Ti = sm.SE3.Trans(action[0], action[1], action[2]) * sm.SE3(sm.SO3(self._T0.R))
                self._robot.move_cartesian(Ti)
                joint_position = self._robot.get_joint()
                self._mj_data.ctrl[:6] = joint_position
                action[3] = np.clip(action[3], 0, 1)
                self._mj_data.ctrl[6] = action[3] * 255.0
                mujoco.mj_step(self._mj_model, self._mj_data)

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
        self._robot_T = self._robot.fkine(self._robot_q)
        agent_pos = np.zeros(4, dtype=np.float32)
        agent_pos[:3] = self._robot_T.t
        agent_pos[3] = np.linalg.norm(self._mj_data.site('left_pad').xpos - self._mj_data.site('right_pad').xpos)

        self._mj_renderer.update_scene(self._mj_data, 0)
        image_top = self._mj_renderer.render()
        # image_top = np.moveaxis(image_top, -1, 0)
        self._mj_renderer.update_scene(self._mj_data, 1)
        image_hand = self._mj_renderer.render()
        # image_hand = np.moveaxis(image_hand, -1, 0)

        obs = {
            'pixels': {
                'top': image_top,
                'hand': image_hand
            },
            'agent_pos': agent_pos
        }
        self._render_cache = image_top
        return obs

    def run(self):
        observation, info = self.reset()

        time0 = 0.04
        T0 = self._robot.get_cartesian()
        t0 = T0.t
        R0 = sm.SO3(T0.R)
        t1 = t0.copy()
        R1 = R0.copy()
        planner0 = self._cal_planner(t0, R0, t1, R1, time0)

        time1 = 2.0
        t2 = t1.copy()
        t2[:] = self._obj_t
        t2[2] = 0.86
        R2 = R1.copy()
        planner1 = self._cal_planner(t1, R1, t2, R2, time1)

        time2 = 1.0
        t3 = t2.copy()
        t3[2] = 0.78
        R3 = R2.copy()
        planner2 = self._cal_planner(t2, R2, t3, R3, time2)

        time3 = 1.0
        t4 = t3.copy()
        R4 = R3.copy()
        planner3 = self._cal_planner(t3, R3, t4, R4, time3)

        time4 = 1.0
        t5 = t4.copy()
        t5[2] = 0.86
        R5 = R4.copy()
        planner4 = self._cal_planner(t4, R4, t5, R5, time4)

        time5 = 2.0
        t6 = np.array([1.4, 0.15, 0.80])
        R6 = R5.copy()
        planner5 = self._cal_planner(t5, R5, t6, R6, time5)

        time6 = 1.0
        t7 = t6.copy()
        R7 = R6.copy()
        planner6 = self._cal_planner(t6, R6, t7, R7, time6)

        time7 = 1.0
        t8 = t7.copy()
        t8[2] = 0.90
        R8 = R7.copy()
        planner7 = self._cal_planner(t7, R7, t8, R8, time7)

        time_array = np.array([time0, time1, time2, time3, time4, time5, time6, time7])
        planner_array = [planner0, planner1, planner2, planner3, planner4, planner5, planner6, planner7]

        observations = []
        actions = []

        time_cumsum = np.cumsum(time_array)

        action = np.zeros(4, dtype=np.float32)
        planner_interpolate = np.zeros(3)
        while True:
            for j in range(len(time_cumsum)):
                if self._mj_data.time <= time_cumsum[j]:
                    if j == 0:
                        start_time = 0.0
                    else:
                        start_time = time_cumsum[j - 1]
                    planner_interpolate = planner_array[j].interpolate(self._mj_data.time - start_time).t
                    break

            else:
                self.close()
                return {
                    "observations": observations,
                    "actions": actions
                }
            action[:3] = planner_interpolate
            if self._mj_data.time >= time_cumsum[5]:
                action[3] = np.maximum(action[3] - 1.0 / time6 / self._control_hz, 0.0)
            elif self._mj_data.time >= time_cumsum[2]:
                action[3] = np.minimum(action[3] + 1.0 / time3 / self._control_hz, 1.0)

            observations.append(observation)
            actions.append(action.copy())

            observation, _, _, _, info = self.step(action)

            self.render()

    def _cal_planner(self, t0, R0, t1, R1, time):
        position_parameter = LinePositionParameter(t0, t1)
        attitude_parameter = OneAttitudeParameter(R0, R1)
        cartesian_parameter = CartesianParameter(position_parameter, attitude_parameter)
        velocity_parameter = QuinticVelocityParameter(time)
        trajectory_parameter = TrajectoryParameter(cartesian_parameter, velocity_parameter)
        trajectory_planner = TrajectoryPlanner(trajectory_parameter)
        return trajectory_planner


if __name__ == '__main__':
    env = PickAndPlaceEnv(render_mode="human")
    env_data = env.run()
