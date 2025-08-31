from dataclasses import dataclass, field

from lerobot.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE
from lerobot.envs import EnvConfig
from lerobot.configs.types import FeatureType, PolicyFeature


@EnvConfig.register_subclass("dishwasher")
@dataclass
class BartendEnvConfig(EnvConfig):
    task: str = "bartend"
    fps: int = 25
    episode_length: int = 10000
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_STATE,
            "overhead_cam": f"{OBS_IMAGE}.overhead_cam",
            "wrist_cam_left": f"{OBS_IMAGE}.wrist_cam_left",
            "wrist_cam_right": f"{OBS_IMAGE}.wrist_cam_right",
            "pixels/overhead_cam": f"{OBS_IMAGES}.overhead_cam",
            "pixels/wrist_cam_left": f"{OBS_IMAGES}.wrist_cam_left",
            "pixels/wrist_cam_right": f"{OBS_IMAGES}.wrist_cam_right",
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels":
            self.features["overhead_cam"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))
            self.features["wrist_cam_left"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))
            self.features["wrist_cam_right"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))
        elif self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(14,))
            self.features["pixels/overhead_cam"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))
            self.features["pixels/wrist_cam_left"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))
            self.features["pixels/wrist_cam_right"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }
