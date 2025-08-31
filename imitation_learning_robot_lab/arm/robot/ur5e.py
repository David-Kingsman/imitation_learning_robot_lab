import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
import modern_robotics as mr

from ..utils import MathUtils
from .robot_config import RobotConfig
from .robot import Robot, get_transformation_mdh, wrap


class UR5e(Robot):

    def __init__(self) -> None:
        super().__init__()
        d1 = 0.163
        d4 = 0.134
        d5 = 0.1
        d6 = 0.1

        a3 = 0.425
        a4 = 0.392

        self._dof = 6
        self.q0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        alpha_array = [0.0, -np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2]
        a_array = [0.0, 0.0, a3, a4, 0.0, 0.0]
        d_array = [d1, 0.0, 0.0, d4, d5, d6]
        theta_array = [0.0, -np.pi / 2, 0.0, np.pi / 2, 0.0, 0.0]
        sigma_array = [0, 0, 0, 0, 0, 0]

        m1 = 3.7
        r1 = np.array([0.0, 0.0, 0.0])
        I1 = np.diag([0.0102675, 0.0102675, 0.00666])
        Jm1 = 0.1

        m2 = 8.393
        r2 = np.array([0.2125, 0.0, 0.138])
        I2 = np.diag([0.0151074, 0.133886, 0.133886])
        Jm2 = 0.1

        m3 = 2.275
        r3 = np.array([0.196, 0.0, 0.007])
        I3 = np.diag([0.004095, 0.0311796, 0.0311796])
        Jm3 = 0.1

        m4 = 1.219
        r4 = np.array([0.0, 0.0, 0.0])
        I4 = np.diag([0.0025599, 0.0021942, 0.0025599])
        Jm4 = 0.1

        m5 = 1.219
        r5 = np.array([0.0, 0.0, 0.0])
        I5 = np.diag([0.0025599, 0.0025599, 0.0021942])
        Jm5 = 0.1

        m6 = 0.1889
        r6 = np.array([0.0, 0.0, -0.0228317])
        I6 = np.diag([9.90863e-05, 9.90863e-05, 0.000132134])
        Jm6 = 0.1

        ms = [m1, m2, m3, m4, m5, m6]
        rs = [r1, r2, r3, r4, r5, r6]
        Is = [I1, I2, I3, I4, I5, I6]
        Jms = [Jm1, Jm2, Jm3, Jm4, Jm5, Jm6]

        links = []
        for i in range(6):
            links.append(rtb.DHLink(d=d_array[i], alpha=alpha_array[i], a=a_array[i], offset=theta_array[i], mdh=True,
                                    m=ms[i], r=rs[i], I=Is[i], Jm=Jms[i], G=1.0))
        self.robot = rtb.DHRobot(links)

        self.alpha_array = alpha_array
        self.a_array = a_array
        self.d_array = d_array
        self.theta_array = theta_array
        self.sigma_array = sigma_array

        T = SE3()
        for i in range(self.dof):
            Ti = get_transformation_mdh(self.alpha_array[i], self.a_array[i], self.d_array[i], self.theta_array[i],
                                        self.sigma_array[i], 0.0)
            self._Ms.append(Ti.A)
            T: SE3 = T * Ti
            self._Ses.append(np.hstack((T.a, np.cross(T.t, T.a))))

            Gm = np.zeros((6, 6))
            Gm[:3, :3] = Is[i]
            Gm[3:, 3:] = ms[i] * np.eye(3)
            AdT = mr.Adjoint(mr.RpToTrans(np.eye(3), -rs[i]))
            self._Gs.append(AdT.T @ Gm @ AdT)
            self._Jms.append(Jms[i])

        self._Ms.append(np.eye(4))

        self.robot_config = RobotConfig()

    def ikine(self, Twt: SE3) -> np.ndarray:

        Tbe = self.cal_Tbe(Twt)

        t_wpc = self.cal_wcp(Tbe)

        thetas = [0.0 for _ in range(self.dof)]

        # solve theta1
        theta1_condition = np.power(t_wpc[1], 2) + np.power(t_wpc[0], 2) - np.power(self.d_array[3], 2)
        if theta1_condition < 0:
            return np.array([])
        if MathUtils.near_zero(np.linalg.norm([t_wpc[1], t_wpc[0]])):
            thetas[0] = self.q0[0] - self.theta_array[0]  # overhead singularity
        else:
            # if self.robot_config.overhead == 0:
            #     thetas[0] = np.arctan2(t_wpc[1], t_wpc[0]) - np.arctan2(self.d_array[3], np.sqrt(theta1_condition))
            # elif self.robot_config.overhead == 1:
            #     thetas[0] = np.arctan2(t_wpc[1], t_wpc[0]) - np.arctan2(self.d_array[3], -np.sqrt(theta1_condition))
            # else:
            #     return np.array([])
            thetas[0] = np.arctan2(t_wpc[1], t_wpc[0]) - np.arctan2(self.d_array[3],
                                                                self.robot_config.overhead * np.sqrt(theta1_condition))

        # solve theta5
        theta5_condition = -Tbe.a[0] * np.sin(thetas[0]) + Tbe.a[1] * np.cos(thetas[0])
        if np.abs(theta5_condition) > 1:
            return np.array([])
        # if self.robot_config.wrist == 0:
        #     thetas[4] = np.arccos(theta5_condition)
        # elif self.robot_config.wrist == 1:
        #     thetas[4] = -np.arccos(theta5_condition)
        # else:
        #     return np.array([])
        thetas[4] = self.robot_config.wrist * np.arccos(theta5_condition)

        # solve theta6
        m1 = - Tbe.n[0] * np.sin(thetas[0]) + Tbe.n[1] * np.cos(thetas[0])
        n1 = - Tbe.o[0] * np.sin(thetas[0]) + Tbe.o[1] * np.cos(thetas[0])
        if MathUtils.near_zero(np.sin(thetas[4])):
            thetas[5] = self.q0[5] - self.theta_array[5]
        else:
            thetas[5] = np.arctan2(-n1 / np.sin(thetas[4]), m1 / np.sin(thetas[4]))

        # solve theta3
        T01 = get_transformation_mdh(self.alpha_array[0], self.a_array[0], self.d_array[0], thetas[0],
                                     self.sigma_array[0],
                                     0.0)
        T45 = get_transformation_mdh(self.alpha_array[4], self.a_array[4], self.d_array[4], thetas[4],
                                     self.sigma_array[4],
                                     0.0)
        T56 = get_transformation_mdh(self.alpha_array[5], self.a_array[5], self.d_array[5], thetas[5],
                                     self.sigma_array[5],
                                     0.0)
        T46 = T45 * T56
        T14 = T01.inv() * Tbe * T46.inv()
        x = T14.t[0]
        y = T14.t[2]

        theta3_condition = (np.power(x, 2) + np.power(y, 2) - np.power(self.a_array[2], 2)
                            - np.power(self.a_array[3], 2)) / (2 * self.a_array[2] * self.a_array[3])
        if np.abs(theta3_condition) > 1.0:
            return np.array([])
        # if self.robot_config.inline == 0:
        #     thetas[2] = np.arccos(theta3_condition)
        # elif self.robot_config.inline == 1:
        #     thetas[2] = - np.arccos(theta3_condition)
        # else:
        #     return np.array([])
        thetas[2] = self.robot_config.inline * np.arccos(theta3_condition)

        # solve theta2
        M = np.array([
            [self.a_array[3] * np.cos(thetas[2]) + self.a_array[2], -self.a_array[3] * np.sin(thetas[2])],
            [self.a_array[3] * np.sin(thetas[2]), self.a_array[3] * np.cos(thetas[2]) + self.a_array[2]]
        ])
        XY = np.array([
            [x],
            [-y]
        ])
        CS = np.linalg.inv(M) @ XY
        thetas[1] = np.arctan2(CS[1, 0], CS[0, 0])

        # solve theta4
        thetas[3] = np.arctan2(-T14.o[0], T14.n[0]) - thetas[1] - thetas[2]

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

        # inline
        if wrap(thetas[2])[0] >= 0:
            self.robot_config.inline = 1
        else:
            self.robot_config.inline = -1

        # wrist
        if wrap(thetas[4])[0] >= 0:
            self.robot_config.wrist = 1
        else:
            self.robot_config.wrist = -1

    def cal_wcp(self, Tbe: SE3) -> np.ndarray:
        t = Tbe.t - self.d_array[5] * Tbe.a
        t[2] -= self.d_array[0]
        return t


if __name__ == '__main__':
    ur_robot = UR5e()
    q0 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    T1 = ur_robot.fkine(q0)
    print(T1)
    ur_robot.move_cartesian(T1)
    q_new = ur_robot.get_joint()
    print(q_new)
