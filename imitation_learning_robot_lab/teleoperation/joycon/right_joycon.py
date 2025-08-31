import time

from pyjoycon import get_R_id, GyroTrackingJoyCon
from imitation_learning_lerobot.teleoperation.joycon.orientation_estimation.imu import Imu


class RightJoycon(GyroTrackingJoyCon, Imu):

    def __init__(self):
        super().__init__(*get_R_id())

    def start(self):
        self.calibrate()
        time.sleep(3)

    def get_gyro(self):
        omega = self.gyro_in_rad[0]
        gx = omega[0] * 2
        gy = -omega[1] * 2
        gz = -omega[2] * 2
        return gx, gy, gz

    def get_acc(self):
        accel = self.accel_in_g[0]
        ax = accel[0]
        ay = -accel[1]
        az = -accel[2]
        return ax, ay, az
