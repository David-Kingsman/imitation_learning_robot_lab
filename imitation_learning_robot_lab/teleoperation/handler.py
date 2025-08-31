import abc

import numpy as np


class Handler(abc.ABC):
    _name = ""

    def __init__(self):
        super().__init__()

        self._action: np.ndarray = None

        self._done = False
        self._sync = False

    @classmethod
    @property
    def name(cls) -> str:
        return cls._name

    def start(self):
        pass

    def close(self):
        pass

    @property
    def action(self):
        return self._action.copy()

    @property
    def done(self):
        return self._done

    @property
    def sync(self):
        return self._sync

    def print_info(self):
        print("------------------------------")
