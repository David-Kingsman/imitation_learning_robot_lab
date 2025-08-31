import threading
import time
import numpy as np
import spatialmath as sm
from loop_rate_limiters import RateLimiter

from ..handler import Handler
from .orientation_estimation import ComplimentaryOrientationEstimation
from .sliding_window_filter import SlidingFilter
from .left_joycon import LeftJoycon
from .right_joycon import RightJoycon


class AlohaJoyconHandler(Handler):
    def __init__(self):
        super().__init__()

        self._timestep = 0.01

        self._state = np.zeros(14)
        self._action = np.zeros(14)
        self._last_action = np.zeros(14)
        self._filter_action = np.zeros(14)

        self._filters = [SlidingFilter(10) for _ in range(14)]

        self._left_joycon = LeftJoycon()
        self._right_joycon = RightJoycon()

        self._left_calibration_offset = np.zeros(8)
        self._right_calibration_offset = np.zeros(8)

        self._left_orientation_estimation = ComplimentaryOrientationEstimation(self._left_joycon)
        self._right_orientation_estimation = ComplimentaryOrientationEstimation(self._right_joycon)

        self._left_sync = False
        self._right_sync = False
        self._done = False

        self._thread: threading.Thread = None
        self._running = True

    def _calibrate(self):
        num_samples = 100
        left_samples = []
        right_samples = []
        for _ in range(num_samples):
            left_status = self._left_joycon.get_status()
            left_accel = left_status['accel']
            left_rot = left_status['gyro']
            left_joystick = left_status['analog-sticks']['left']

            right_status = self._right_joycon.get_status()
            right_accel = right_status['accel']
            right_rot = right_status['gyro']
            right_joystick = right_status['analog-sticks']['right']

            left_samples.append(
                [left_accel['x'], left_accel['y'], left_accel['z'], left_rot['x'], left_rot['y'], left_rot['z'],
                 left_joystick['horizontal'], left_joystick['vertical']])

            right_samples.append(
                [right_accel['x'], right_accel['y'], right_accel['z'], right_rot['x'], right_rot['y'], right_rot['z'],
                 right_joystick['horizontal'], right_joystick['vertical']])
            time.sleep(0.01)

        self._left_calibration_offset = np.mean(left_samples, axis=0)
        self._right_calibration_offset = np.mean(right_samples, axis=0)

    def start(self):
        time.sleep(1.0)
        self._left_orientation_estimation.start()
        self._right_orientation_estimation.start()

        self._calibrate()

        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def _update_loop(self):
        rate_limiter = RateLimiter(frequency=1.0 / self._timestep)
        while self._running:
            self._left_update()
            self._right_update()
            rate_limiter.sleep()

    def _left_update(self):
        status = self._left_joycon.get_status()
        if not self._left_sync:
            if status['buttons']['left']['left']:
                self._state[:3] = 0
                self._state[3:6] = self._left_orientation_estimation.euler_angles
                self._last_action[3:6] = self._action[3:6]
                self._left_sync = True
        else:
            if status['buttons']['left']['right']:
                self._left_sync = False
        if status['buttons']['left']['sl']:
            self._done = True

        if not self._left_sync:
            return

        button_lower = status['buttons']['left']['zl']
        button_higher = status['buttons']['left']['l']
        joystick = status['analog-sticks']['left']
        up = status['buttons']['left']['up']
        down = status['buttons']['left']['down']

        R0 = sm.SO3.RPY(self._state[3:6])
        R_global = sm.SO3.RPY(self._left_orientation_estimation.euler_angles)
        delta_R: sm.SO3 = R0.inv() * R_global

        R_env = sm.SO3.RPY(self._last_action[3:6]) * delta_R
        trans = np.zeros(3)
        trans[0] = (joystick['vertical'] - self._left_calibration_offset[7]) * 0.000001
        trans[1] = -(joystick['horizontal'] - self._left_calibration_offset[6]) * 0.000001
        trans[2] = 0.001 if button_higher == 1 else -0.001 if button_lower == 1 else 0

        left_action = self._action[:7].copy()
        left_action[0: 3] += (R_env * trans).flatten()
        left_action[3: 6] = (sm.SO3.RPY(self._last_action[3:6]) * delta_R).rpy()
        left_action[6] += 0.01 if up == 1 else -0.01 if down == 1 else 0
        left_action[6] = np.clip(left_action[6], 0.0, 1.0)

        self._action[:7] = left_action

        filter_action = np.zeros(7)
        for i in range(7):
            filter_action[i] = self._filters[i].add_sample(left_action[i])

        self._filter_action[:7] = filter_action

    def _right_update(self):
        status = self._right_joycon.get_status()
        if not self._right_sync:
            if status['buttons']['right']['a']:
                self._state[7:10] = 0
                self._state[10:13] = self._right_orientation_estimation.euler_angles
                self._last_action[10:13] = self._action[10:13]
                self._right_sync = True
        else:
            if status['buttons']['right']['y']:
                self._right_sync = False
        if status['buttons']['right']['sr']:
            self._done = True

        if not self._right_sync:
            return

        button_lower = status['buttons']['right']['zr']
        button_higher = status['buttons']['right']['r']
        joystick = status['analog-sticks']['right']
        up = status['buttons']['right']['x']
        down = status['buttons']['right']['b']

        R0 = sm.SO3.RPY(self._state[10:13])
        R_global = sm.SO3.RPY(self._right_orientation_estimation.euler_angles)
        delta_R: sm.SO3 = R0.inv() * R_global

        R_env = sm.SO3.RPY(self._last_action[10:13]) * delta_R
        trans = np.zeros(3)
        trans[0] = (joystick['vertical'] - self._right_calibration_offset[7]) * 0.000001
        trans[1] = -(joystick['horizontal'] - self._right_calibration_offset[6]) * 0.000001
        trans[2] = 0.001 if button_higher == 1 else -0.001 if button_lower == 1 else 0

        right_action = self._action[7:].copy()
        right_action[:3] += (R_env * trans).flatten()
        right_action[3:6] = (sm.SO3.RPY(self._last_action[10:13]) * delta_R).rpy()
        right_action[6] += 0.01 if up == 1 else -0.01 if down == 1 else 0
        right_action[6] = np.clip(right_action[6], 0.0, 1.0)

        self._action[7:] = right_action

        filter_action = np.zeros(7)
        for i in range(7):
            filter_action[i] = self._filters[i + 7].add_sample(right_action[i])

        self._filter_action[7:14] = filter_action

    def close(self):
        self._left_orientation_estimation.close()
        self._right_orientation_estimation.close()

        self._running = False
        self._thread.join()

    @property
    def action(self):
        return self._filter_action.copy()

    @property
    def done(self):
        return self._done

    @property
    def sync(self):
        return self._left_sync or self._right_sync

    def print_info(self):
        print("----------------------------------------------------------------")
        print("             Left Joy-Con                    Right Joy-Con")
        print("Start:       Left(Button)                    A(Button)")
        print("Pause:       Right(Button)                   Y(Button)")
        print("Stop:        SL                              SR")
        print("Forward:     Up(Joystick)                    Up(Joystick)")
        print("Backward:    Down(Joystick)                  Down(Joystick)")
        print("Leftward:    Left(Joystick)                  Left(Joystick)")
        print("Rightward:   Right(Joystick)                 Right(Joystick)")
        print("Upward:      L                               R")
        print("Downward:    ZL                              ZR")
        print("Open:        Down(Button)                    B(Button)")
        print("Close:       UP(Button)                      X(Button)")

