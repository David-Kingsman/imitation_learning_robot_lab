import time

import numpy as np

from pyjoycon import GyroTrackingJoyCon, get_R_id
from sympy.benchmarks.bench_meijerint import alpha

from .imu import Imu
from .orientation_estimation import OrientationEstimation


class ComplimentaryOrientationEstimation(OrientationEstimation):

    def __init__(self, imu: Imu):
        super().__init__(imu)

        self._alpha = 0.1

    def update(self):
        phi_hat = self._phi
        theta_hat = self._theta
        psi_hat = self._psi

        ax, ay, az = self._imu.get_acc()
        gx, gy, gz = self._imu.get_gyro()

        phi_hat_acc = np.arctan2(ay, az)
        theta_hat_acc = np.arctan2(-ax, np.linalg.norm([ay, az]))
        self._psi = 0.0

        phi_dot = gx + np.sin(phi_hat) * np.tan(theta_hat) * gy + np.cos(phi_hat) * np.tan(theta_hat) * gz
        theta_dot = np.cos(phi_hat) * gy - np.sin(phi_hat) * gz
        psi_dot = (np.sin(phi_hat) * gy + np.cos(phi_hat) * gz) / np.cos(theta_hat)

        self._phi = (1 - self._alpha) * (phi_hat + self._timestep * phi_dot) + self._alpha * phi_hat_acc
        self._theta = (1 - self._alpha) * (theta_hat + self._timestep * theta_dot) + self._alpha * theta_hat_acc
        self._psi = psi_hat + self._timestep * psi_dot
