import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
import modern_robotics as mr

from ..utils import MathUtils
from .robot_config import RobotConfig
from .robot import Robot, get_transformation_mdh, wrap


class VX300S(Robot):
    def __init__(self) -> None:
        super().__init__()
        d1 = 0.079 + 0.04805
        d3 = 0.0
        d4 = 0.2 + 0.1
        d6 = 0.069744

        a2 = 0.0
        a3 = np.linalg.norm([0.05955, 0.3])
        a4 = 0.0

        theta2_offset = np.arctan2(0.05955, 0.3)

        self._dof = 6
        self.q0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        alpha_array = [0.0, -np.pi / 2, 0.0, -np.pi / 2, np.pi / 2, -np.pi / 2]
        a_array = [0.0, a2, a3, a4, 0.0, 0.0]
        d_array = [d1, 0.0, d3, d4, 0.0, d6]
        theta_array = [0.0, -np.pi / 2 + theta2_offset, -theta2_offset, 0.0, 0.0, 0.0]
        sigma_array = [0, 0, 0, 0, 0, 0]

        links = []
        for i in range(self._dof):
            links.append(rtb.DHLink(d=d_array[i], alpha=alpha_array[i], a=a_array[i], offset=theta_array[i], mdh=True))
        self.robot = rtb.DHRobot(links)

        self.alpha_array = alpha_array
        self.a_array = a_array
        self.d_array = d_array
        self.theta_array = theta_array
        self.sigma_array = sigma_array

        self._direction = np.array([1, 1, 1, 1, 1, 1])

        T = SE3()
        for i in range(self.dof):
            Ti = get_transformation_mdh(self.alpha_array[i], self.a_array[i], self.d_array[i], self.theta_array[i],
                                        self.sigma_array[i], 0.0)
            self._Ms.append(Ti.A)
            T: SE3 = T * Ti
            self._Ses.append(np.hstack((T.a, np.cross(T.t, T.a))))

            # Gm = np.zeros((6, 6))
            # Gm[:3, :3] = Is[i]
            # Gm[3:, 3:] = ms[i] * np.eye(3)
            # AdT = mr.Adjoint(mr.RpToTrans(np.eye(3), -rs[i]))
            # self._Gs.append(AdT.T @ Gm @ AdT)
            # self._Jms.append(Jms[i])

        self._Ms.append(np.eye(4))

        self.robot_config = RobotConfig()

    def ikine(self, Twt: SE3) -> np.ndarray:

        Tbe = self.cal_Tbe(Twt)

        t_wcp = self.cal_wcp(Tbe)

        thetas = np.zeros(self._dof)

        theta1_condition = np.power(t_wcp[1], 2) + np.power(t_wcp[0], 2) - np.power(self.d_array[2], 2)
        if theta1_condition < 0:
            return np.array([])
        if MathUtils.near_zero(np.linalg.norm([t_wcp[1], t_wcp[0]])):
            thetas[0] = self.q0[0] - self.theta_array[0]
        else:
            thetas[0] = np.arctan2(t_wcp[1], t_wcp[0]) - np.arctan2(self.d_array[2],
                                                                self.robot_config.overhead * np.sqrt(theta1_condition))

        k3 = (np.power(self.a_array[1] - np.cos(thetas[0]) * t_wcp[0] - np.sin(thetas[0]) * t_wcp[1], 2) + np.power(t_wcp[2],
                                                                                                                2) - (
                      np.power(self.a_array[3], 2) + np.power(self.d_array[3], 2) + np.power(self.a_array[2],
                                                                                             2))) / (
                     2 * self.a_array[2])
        theta3_condition = np.power(self.a_array[3], 2) + np.power(self.d_array[3], 2) - np.power(k3, 2)
        if theta3_condition < 0:
            return np.array([])
        thetas[2] = np.arctan2(self.a_array[3], self.d_array[3]) - np.arctan2(k3, self.robot_config.inline * np.sqrt(
            theta3_condition))

        g = np.cos(thetas[0]) * t_wcp[0] + np.sin(thetas[0]) * t_wcp[1] - self.a_array[1]
        e = self.a_array[3] * np.cos(thetas[2]) - self.d_array[3] * np.sin(thetas[2]) + self.a_array[2]
        f = -(self.a_array[3] * np.cos(thetas[2]) + self.d_array[3] * np.cos(thetas[2]))
        thetas[1] = np.arctan2(g * f - t_wcp[2] * e, g * e + t_wcp[2] * f)

        T01 = get_transformation_mdh(self.alpha_array[0], self.a_array[0], self.d_array[0], thetas[0],
                                     self.sigma_array[0], 0.0)
        T12 = get_transformation_mdh(self.alpha_array[1], self.a_array[1], self.d_array[1], thetas[1],
                                     self.sigma_array[1], 0.0)
        T23 = get_transformation_mdh(self.alpha_array[2], self.a_array[2], self.d_array[2], thetas[2],
                                     self.sigma_array[2], 0.0)

        T03: SE3 = T01 * T12 * T23
        T36: SE3 = T03.inv() * Tbe

        thetas[4] = self.robot_config.wrist * np.arccos(T36.a[1])

        thetas[3] = np.arctan2(T36.a[2] / np.sin(thetas[4]), -T36.a[0] / np.sin(thetas[4]))
        thetas[5] = np.arctan2(-T36.o[1] / np.sin(thetas[4]), T36.n[1] / np.sin(thetas[4]))

        qs = np.array([wrap(thetas[i] - self.theta_array[i])[0] for i in range(self.dof)])

        q0_s = list(map(wrap, self.q0))

        for i in range(self.dof):
            if qs[i] - q0_s[i][0] > np.pi:
                qs[i] += (q0_s[i][1] - 1) * 2 * np.pi
            elif qs[i] - q0_s[i][0] < -np.pi:
                qs[i] += (q0_s[i][1] + 1) * 2 * np.pi
            else:
                qs[i] += q0_s[i][1] * 2 * np.pi

        return qs

    def set_robot_config(self, q):
        Twt = self.fkine(q)
        Tbe = self.cal_Tbe(Twt)
        t_wpc = self.cal_wcp(Tbe)
        thetas = [q[i] + self.theta_array[i] for i in range(self.dof)]

        if np.cos(wrap(np.arctan2(t_wpc[1], t_wpc[0]) - wrap(thetas[0])[0])[0]) >= 0.0:
            self.robot_config.overhead = 1
        else:
            self.robot_config.overhead = -1

        if np.cos(wrap(wrap(thetas[2])[0] + np.arctan2(self.a_array[3], self.d_array[3]))[0]) >= 0.0:
            self.robot_config.inline = 1
        else:
            self.robot_config.inline = -1

        if wrap(thetas[4])[0] >= 0.0:
            self.robot_config.wrist = 1
        else:
            self.robot_config.wrist = -1

    def cal_wcp(self, Tbe: SE3) -> np.ndarray:
        t = Tbe.t - self.d_array[5] * Tbe.a
        t[2] -= self.d_array[0]
        return t

    def set_joint(self, q):
        super().set_joint(self._direction * q)

    def get_joint(self):
        return super().get_joint() * self._direction
