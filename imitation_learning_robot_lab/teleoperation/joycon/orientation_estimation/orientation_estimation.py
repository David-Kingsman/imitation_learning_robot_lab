import abc
import threading

import numpy as np
from loop_rate_limiters import RateLimiter

from .imu import Imu


class OrientationEstimation(abc.ABC):

    def __init__(self, imu: Imu):
        super().__init__()

        self._phi = 0.0
        self._theta = 0.0
        self._psi = 0.0
        self._timestep = 0.01
        self._rate_limiter = RateLimiter(frequency=1.0 / self._timestep)

        self._imu = imu

        self._thread: threading.Thread = None
        self._running = True

    def start(self):
        self._imu.start()

        self._thread = threading.Thread(target=self.update_loop, daemon=True)
        self._thread.start()

    def close(self):
        self._running = False
        self._thread.join()

    def update_loop(self):
        while self._running:
            self.update()
            self._rate_limiter.sleep()

    @abc.abstractmethod
    def update(self):
        pass

    @property
    def euler_angles(self):
        return self._phi, self._theta, self._psi
