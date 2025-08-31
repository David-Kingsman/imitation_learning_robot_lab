import threading
import time
import numpy as np
from loop_rate_limiters import RateLimiter

from .right_joycon import RightJoycon
from ..handler import Handler


class PickBoxJoyconHandler(Handler):
    _name = "pick_box_joycon"

    def __init__(self):
        super().__init__()

        self._timestep = 0.01

        self._action = np.zeros(4)

        self._right_joycon = RightJoycon()

        self._right_calibration_offset = np.zeros(8)

        self._sync = False
        self._done = False

        self._thread: threading.Thread = None
        self._running = True

    def _calibrate(self):
        num_samples = 100
        right_samples = []
        for _ in range(num_samples):
            right_status = self._right_joycon.get_status()
            right_accel = right_status['accel']
            right_rot = right_status['gyro']
            right_joystick = right_status['analog-sticks']['right']

            right_samples.append(
                [right_accel['x'], right_accel['y'], right_accel['z'], right_rot['x'], right_rot['y'], right_rot['z'],
                 right_joystick['horizontal'], right_joystick['vertical']])
            time.sleep(0.01)

        self._right_calibration_offset[:] = np.mean(right_samples, axis=0)

    def start(self):
        time.sleep(1.0)
        self._calibrate()

        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def _update_loop(self):
        rate_limiter = RateLimiter(frequency=1.0 / self._timestep)
        while self._running:
            self._right_update()
            rate_limiter.sleep()

    def _right_update(self):
        status = self._right_joycon.get_status()
        if not self._sync:
            if status['buttons']['right']['a']:
                self._sync = True
        else:
            if status['buttons']['right']['y']:
                self._sync = False
        if status['buttons']['right']['sr']:
            self._done = True

        if not self._sync:
            return

        button_lower = status['buttons']['right']['zr']
        button_higher = status['buttons']['right']['r']
        joystick = status['analog-sticks']['right']
        up = status['buttons']['right']['x']
        down = status['buttons']['right']['b']

        self._action[0] -= (joystick['horizontal'] - self._right_calibration_offset[6]) * 0.000002
        self._action[1] += (joystick['vertical'] - self._right_calibration_offset[7]) * 0.000002
        self._action[2] += 0.002 if button_lower == 1 else -0.002 if button_higher == 1 else 0
        self._action[3] += 0.01 if up == 1 else -0.01 if down == 1 else 0.0
        self._action[3] = np.clip(self._action[3], 0.0, 1.0)

    def close(self):
        self._running = False
        self._thread.join()

    def print_info(self):
        print("------------------------------")
        print("Start:           A")
        print("Pause:           Y")
        print("Stop:            SR")
        print("+X:              Down")
        print("-X:              Up")
        print("+Y:              Right")
        print("-Y:              Left")
        print("+Z:              R")
        print("-Z:              ZR")
        print("Open:            B")
        print("Close:           X")
