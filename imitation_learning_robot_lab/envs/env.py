import abc


class Env(abc.ABC):
    _name = ""
    _robot_type = ""
    _height = 480
    _width = 640
    _states = []
    _cameras = []
    _state_dim = 0
    _action_dim = 0
    _control_hz = 25

    def __init__(self, render_mode: str = "rgb_array"):
        pass

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass

    def _get_observation(self):
        pass

    def run(self):
        pass

    @classmethod
    @property
    def name(cls) -> str:
        return cls._name

    @classmethod
    @property
    def robot_type(cls) -> str:
        return cls._robot_type

    @classmethod
    @property
    def height(cls) -> int:
        return cls._height

    @classmethod
    @property
    def width(cls) -> int:
        return cls._width

    @classmethod
    @property
    def states(cls) -> list[str]:
        return cls._states

    @classmethod
    @property
    def cameras(cls) -> list[str]:
        return cls._cameras

    @classmethod
    @property
    def state_dim(cls):
        return cls._state_dim

    @classmethod
    @property
    def action_dim(cls):
        return cls._action_dim

    @classmethod
    @property
    def control_hz(cls) -> int:
        return cls._control_hz
