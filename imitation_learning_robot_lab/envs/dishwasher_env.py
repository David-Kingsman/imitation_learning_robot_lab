import os
import time
from pathlib import Path
import numpy as np
import spatialmath as sm

import mujoco
import mujoco.viewer

from .env import Env

from ..arm.robot import Robot, VX300S
from ..arm.motion_planning import LinePositionParameter, OneAttitudeParameter, \
    CartesianParameter, \
    QuinticVelocityParameter, TrajectoryParameter, TrajectoryPlanner
from ..utils import mj


class DishwasherEnv(Env):
    _name = "dishwasher"
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

        scene_path = Path(__file__).parent.parent / Path("assets/aloha/scene.xml")
        self._mj_model: mujoco.MjModel = mujoco.MjModel.from_xml_path(os.fspath(scene_path))
        self._mj_data: mujoco.MjData = mujoco.MjData(self._mj_model)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._left_robot = VX300S()
        self._left_robot_q = np.zeros(self._left_robot.dof)
        self._left_robot_joint_names = ["left/waist", "left/shoulder", "left/elbow", "left/forearm_roll",
                                        "left/wrist_angle", "left/wrist_rotate"]
        self._left_robot_T = sm.SE3()
        self._left_T0 = sm.SE3()
        self._left_tool_joint_name = "left/left_finger"

        self._right_robot = VX300S()
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
        mj.set_joint_q(self._mj_model, self._mj_data, "left/left_finger", 0.037)
        mj.set_joint_q(self._mj_model, self._mj_data, "left/right_finger", 0.037)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._left_robot.set_tool(sm.SE3.RPY(0.0, -np.pi / 2, np.pi) * sm.SE3.Trans([0.13, 0.0, -0.003]))
        self._left_robot_T = self._left_robot.fkine(self._left_robot_q)
        self._left_T0 = self._left_robot_T.copy()

        self._right_robot.set_base(mj.get_body_pose(self._mj_model, self._mj_data, "right/base_link"))
        self._right_robot_q = np.array([0.0, -0.5, 0.0, 0.0, 0.5, 0.0])
        self._right_robot.set_joint(self._right_robot_q)
        [mj.set_joint_q(self._mj_model, self._mj_data, jn, self._right_robot_q[i]) for i, jn in
         enumerate(self._right_robot_joint_names)]
        mj.set_joint_q(self._mj_model, self._mj_data, "right/left_finger", 0.037)
        mj.set_joint_q(self._mj_model, self._mj_data, "right/right_finger", 0.037)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._right_robot.set_tool(sm.SE3.RPY(0.0, -np.pi / 2, np.pi) * sm.SE3.Trans([0.13, 0.0, -0.003]))
        self._right_robot_T = self._right_robot.fkine(self._right_robot_q)
        self._right_T0 = self._right_robot_T.copy()

        mj_ctrl = np.zeros(14)
        mj_ctrl[:6] = self._left_robot_q
        mj_ctrl[6] = 0.037
        mj_ctrl[7:13] = self._right_robot_q
        mj_ctrl[13] = 0.037
        mujoco.mj_setState(self._mj_model, self._mj_data, mj_ctrl, mujoco.mjtState.mjSTATE_CTRL)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        px_dish_drainer = np.random.uniform(-0.15, -0.05)
        py_dish_drainer = -0.2
        pz_dish_drainer = 0.0
        rz_dish_drainer = np.random.uniform(-1.0, 0.2)
        T_dish_drainer = sm.SE3.Rt(R=sm.SO3.Rz(rz_dish_drainer),
                                   t=np.array([px_dish_drainer, py_dish_drainer, pz_dish_drainer]))

        dish_drainer_eq_data = np.zeros(11)
        dish_drainer_eq_data[3:6] = T_dish_drainer.t
        dish_drainer_eq_data[6:10] = T_dish_drainer.UnitQuaternion()
        dish_drainer_eq_data[-1] = 1.0
        mj.attach(self._mj_model, self._mj_data, "dish_drainer_attach",
                  "dish_drainer_free_joint", T_dish_drainer, eq_data=dish_drainer_eq_data)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        px_plate = px_dish_drainer
        py_plate = py_dish_drainer
        pz_plate = 0.15
        randint_index_plate = np.random.randint(-3, 3)
        T_plate = sm.SE3.Rt(R=sm.SO3.RPY(-np.pi / 2, 0.0, rz_dish_drainer),
                            t=np.array([px_plate, py_plate, pz_plate]) + (
                                    0.042 * randint_index_plate + 0.015) * T_dish_drainer.o)
        mj.set_free_joint_pose(self._mj_model, self._mj_data, "plate_free_joint", T_plate)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._mj_renderer = mujoco.renderer.Renderer(self._mj_model, height=self._height, width=self._width)
        if self._render_mode == "human":
            self._mj_viewer = mujoco.viewer.launch_passive(self._mj_model, self._mj_data)

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
            self._mj_data.ctrl[6] = action[6] * (0.002 - 0.037) + 0.037

            right_Ti = self._right_T0 * sm.SE3.Rt(R=sm.SO3.RPY(action[10], action[11], action[12]), t=action[7:10])
            self._right_robot.move_cartesian(right_Ti)
            right_joint_position = self._right_robot.get_joint()
            self._mj_data.ctrl[7:13] = right_joint_position
            action[13] = np.clip(action[13], 0, 1)
            self._mj_data.ctrl[13] = action[13] * (0.002 - 0.037) + 0.037

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
        observation, info = self.reset()

        observations = []
        actions = []

        time0 = 0.04
        left_T0 = self._left_robot.get_cartesian()
        left_t0 = left_T0.t
        left_R0 = sm.SO3(left_T0.R)
        left_t1 = left_t0.copy()
        left_R1 = left_R0.copy()
        left_planner0 = self._cal_planner(left_t0, left_R0, left_t1, left_R1, time0)

        right_T0 = self._right_robot.get_cartesian()
        right_t0 = right_T0.t
        right_R0 = sm.SO3(right_T0.R)
        right_t1 = right_t0.copy()
        right_R1 = right_R0.copy()
        right_planner0 = self._cal_planner(right_t0, right_R0, right_t1, right_R1, time0)

        T_plate = mj.get_body_pose(self._mj_model, self._mj_data, "plate")

        n_left_tool = np.array([1.0, 0.0, -1.0])
        n_left_tool[:] = n_left_tool / np.linalg.norm(n_left_tool)
        o_left_tool = np.sign(np.dot(T_plate.a, np.array([0.0, 1.0, 0.0]))) * T_plate.a
        a_left_tool = np.cross(n_left_tool, o_left_tool)
        R_left_tool = sm.SO3.TwoVectors(y=o_left_tool, z=a_left_tool)

        time1 = 2.0
        left_t2 = T_plate.t - 0.15 * R_left_tool.n + 0.01 * T_plate.a
        left_R2 = R_left_tool
        left_planner1 = self._cal_planner(left_t1, left_R1, left_t2, left_R2, time1)

        right_t2 = np.array([0.05, 0.32, 0.31])
        right_R2 = sm.SO3.Rz(np.pi) * sm.SO3.Ry(np.pi / 4)
        right_planner1 = self._cal_planner(right_t1, right_R1, right_t2, right_R2, time1)

        time2 = 1.0
        left_t3 = left_t2 + left_R2.n * 0.11
        left_R3 = left_R2.copy()
        left_planner2 = self._cal_planner(left_t2, left_R2, left_t3, left_R3, time2)

        right_t3 = right_t2.copy()
        right_t3[2] -= 0.1
        right_R3 = right_R2.copy()
        right_planner2 = self._cal_planner(right_t2, right_R2, right_t3, right_R3, time2)

        time3 = 1.0
        left_t4 = left_t3.copy()
        left_R4 = left_R3.copy()
        left_planner3 = self._cal_planner(left_t3, left_R3, left_t4, left_R4, time3)

        right_t4 = right_t3.copy()
        right_R4 = right_R3.copy()
        right_planner3 = self._cal_planner(right_t3, right_R3, right_t4, right_R4, time3)

        time4 = 2.0
        left_t5 = left_t4.copy()
        left_t5[2] += 0.2
        left_R5 = left_R4.copy()
        left_planner4 = self._cal_planner(left_t4, left_R4, left_t5, left_R5, time4)

        right_t5 = right_t4.copy()
        right_R5 = right_R4.copy()
        right_planner4 = self._cal_planner(right_t4, right_R4, right_t5, right_R5, time4)

        time5 = 2.0
        left_t6 = np.array([-0.2, 0.0, 0.55])
        left_R6 = sm.SO3.TwoVectors(x=np.array([1.0, 0.0, -0.4]), z=np.array([0.0, -1.0, 0.0]))
        left_planner5 = self._cal_planner(left_t5, left_R5, left_t6, left_R6, time5)

        right_t6 = right_t5.copy()
        right_R6 = right_R5.copy()
        right_planner5 = self._cal_planner(right_t5, right_R5, right_t6, right_R6, time5)

        time6 = 2.0
        left_t7 = np.array([-0.1, 0.120, 0.50])
        left_R7 = left_R6.copy()
        left_planner6 = self._cal_planner(left_t6, left_R6, left_t7, left_R7, time6)

        right_t7 = right_t6.copy()
        right_t7[1] -= 0.35
        right_R7 = right_R6.copy()
        right_planner6 = self._cal_planner(right_t6, right_R6, right_t7, right_R7, time6)

        time7 = 2.0
        left_t8 = left_t7.copy()
        left_t8[2] = 0.35
        left_R8 = sm.SO3.TwoVectors(x=np.array([0.0, 0.0, -1.0]), z=np.array([0.0, -1.0, 0.0]))
        left_planner7 = self._cal_planner(left_t7, left_R7, left_t8, left_R8, time7)

        right_t8 = right_t7.copy()
        right_R8 = right_R7.copy()
        right_planner7 = self._cal_planner(right_t7, right_R7, right_t8, right_R8, time7)

        time8 = 2.0
        left_t9 = left_t8.copy()
        left_t9[2] = 0.21
        left_R9 = left_R8.copy()
        left_planner8 = self._cal_planner(left_t8, left_R8, left_t9, left_R9, time8)

        right_t9 = right_t8.copy()
        right_R9 = right_R8.copy()
        right_planner8 = self._cal_planner(right_t8, right_R8, right_t9, right_R9, time8)

        time9 = 1.0
        left_t10 = left_t9.copy()
        left_R10 = left_R9.copy()
        left_planner9 = self._cal_planner(left_t9, left_R9, left_t10, left_R10, time9)

        right_t10 = right_t9.copy()
        right_R10 = right_R9.copy()
        right_planner9 = self._cal_planner(right_t9, right_R9, right_t10, right_R10, time9)

        time10 = 1.0
        left_t11 = left_t10.copy() - 0.1 * left_R10.n
        left_R11 = left_R10.copy()
        left_planner10 = self._cal_planner(left_t10, left_R10, left_t11, left_R11, time10)

        right_t11 = right_t10.copy()
        right_R11 = right_R10.copy()
        right_planner10 = self._cal_planner(right_t10, right_R10, right_t11, right_R11, time10)

        time11 = 2.0
        left_t12 = left_t11.copy()
        left_R12 = left_R11.copy()
        left_planner11 = self._cal_planner(left_t11, left_R11, left_t12, left_R12, time11)

        right_t12 = right_t11.copy()
        right_t12[1] += 0.35
        right_R12 = right_R11.copy()
        right_planner11 = self._cal_planner(right_t11, right_R11, right_t12, right_R12, time11)

        time12 = 1.0
        left_t13 = left_t12.copy()
        left_R13 = left_R12.copy()
        left_planner12 = self._cal_planner(left_t12, left_R12, left_t13, left_R13, time12)

        right_t13 = right_t12.copy()
        right_R13 = right_R12.copy()
        right_planner12 = self._cal_planner(right_t12, right_R12, right_t13, right_R13, time12)

        time13 = 1.0
        left_t14 = left_t13.copy()
        left_R14 = left_R13.copy()
        left_planner13 = self._cal_planner(left_t13, left_R13, left_t14, left_R14, time13)

        right_t14 = right_t13.copy()
        right_t14[2] += 0.125
        right_R14 = right_R13.copy()
        right_planner13 = self._cal_planner(right_t13, right_R13, right_t14, right_R14, time13)

        time14 = 2.0
        left_t15 = self._left_T0.t.copy()
        left_R15 = sm.SO3(self._left_T0)
        left_planner14 = self._cal_planner(left_t14, left_R14, left_t15, left_R15, time14)

        right_t15 = self._right_T0.t.copy()
        right_R15 = sm.SO3(self._right_T0)
        right_planner14 = self._cal_planner(right_t14, right_R14, right_t15, right_R15, time14)

        time_array = np.array([time0, time1, time2, time3, time4, time5, time6, time7, time8, time9,
                               time10, time11, time12, time13, time14])
        left_planner_array = [left_planner0, left_planner1, left_planner2, left_planner3,
                              left_planner4, left_planner5, left_planner6, left_planner7,
                              left_planner8, left_planner9, left_planner10, left_planner11,
                              left_planner12, left_planner13, left_planner14]
        right_planner_array = [right_planner0, right_planner1, right_planner2, right_planner3,
                               right_planner4, right_planner5, right_planner6, right_planner7,
                               right_planner8, right_planner9, right_planner10, right_planner11,
                               right_planner12, right_planner13, right_planner14]

        time_cumsum = np.cumsum(time_array)
        left_planner_interpolate = sm.SE3()
        right_planner_interpolate = sm.SE3()
        action = np.zeros(14, dtype=np.float32)

        while True:

            for j in range(len(time_cumsum)):
                if (self._mj_data.time - self._ready_time) <= time_cumsum[j]:
                    if j == 0:
                        start_time = 0.0
                    else:
                        start_time = time_cumsum[j - 1]
                    left_planner_interpolate = left_planner_array[j].interpolate(
                        self._mj_data.time - self._ready_time - start_time)
                    right_planner_interpolate = right_planner_array[j].interpolate(
                        self._mj_data.time - self._ready_time - start_time)
                    break

            else:
                self.close()
                return {
                    "observations": observations,
                    "actions": actions
                }

            local_left_planner_interpolate: sm.SE3 = self._left_T0.inv() * left_planner_interpolate
            local_right_planner_interpolate: sm.SE3 = self._right_T0.inv() * right_planner_interpolate

            action[:3] = local_left_planner_interpolate.t
            action[3:6] = local_left_planner_interpolate.rpy()
            action[7:10] = local_right_planner_interpolate.t
            action[10:13] = local_right_planner_interpolate.rpy()
            if (self._mj_data.time - self._ready_time) <= time_cumsum[2]:
                action[6] = 1.0
                action[13] = 1.0
            elif (self._mj_data.time - self._ready_time) <= time_cumsum[8]:
                action[6] = np.maximum(action[6] - 1.0 / time9 / self._control_hz, 0.0)
                action[13] = np.maximum(action[13] - 1.0 / time9 / self._control_hz, 0.0)
            elif (self._mj_data.time - self._ready_time) <= time_cumsum[11]:
                action[6] = np.minimum(action[6] + 1.0 / time12 / self._control_hz, 1.0)
                action[13] = 0.0
            else:
                action[6] = np.minimum(action[6] + 1.0 / time13 / self._control_hz, 1.0)
                action[13] = np.minimum(action[13] + 1.0 / time13 / self._control_hz, 1.0)

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
    env = DishwasherEnv(render_mode="human")
    env.run()
