from typing import Type, Optional
from pathlib import Path
import argparse
import dataclasses
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from imitation_learning_lerobot.envs import Env, EnvFactory

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    # video_backend: str | None = None
    video_backend: Optional[str] = "h264"  # use h264 encoder to improve compatibility

# args parser
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--env.type',
        type=str,
        dest='env_type',
        required=True,
        help='env type'
    )

    parser.add_argument(
        '--episode',
        type=int,
        default=100,
        help='episode'
    )
    return parser.parse_args()

# create empty dataset
def create_empty_dataset(env_cls: Type[Env]):
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(env_cls.states),),
            "names": {
                "position": env_cls.states,
            }
        }, "action": {
            "dtype": "float32",
            "shape": (len(env_cls.states),),
            "names": {
                "position": env_cls.states,
            }
        }
    }
    # add video features
    for camera in env_cls.cameras:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (env_cls.height, env_cls.width, 3),
            "names": [
                "height",
                "width",
                "channel"
            ]
        }
    # create dataset
    config = DatasetConfig()
    dataset = LeRobotDataset.create(
        repo_id=env_cls.name,
        fps=env_cls.control_hz,
        features=features,
        root=Path(__file__).parent.parent.parent / Path("outputs/datasets") / Path(env_cls.name),
        robot_type=env_cls.robot_type,
        use_videos=config.use_videos,
        tolerance_s=config.tolerance_s,
        image_writer_processes=config.image_writer_processes,
        image_writer_threads=config.image_writer_threads,
        # video_backend=config.video_backend
        video_backend="h264"  # use h264 encoder to improve compatibility
    )

    return dataset

# populate dataset
def populate_dataset(episode: int, env_cls: Type[Env], dataset: LeRobotDataset):
    env = env_cls()
    task = env.name
    for i in range(episode):
        data = env.run()

        episode_length = len(data["observations"])

        for j in range(episode_length):
            frame = {
                "observation.state": data["observations"][j]["agent_pos"],
                "action": data["actions"][j],
            }

            for camera in env_cls.cameras:
                frame[f"observation.images.{camera}"] = data["observations"][j]["pixels"][camera]

            dataset.add_frame(frame, task=task)
        dataset.save_episode()

    env.close()

# main
def main():
    args = parse_args()

    env_type = args.env_type
    env_cls = EnvFactory.get_strategies(env_type)

    dataset = create_empty_dataset(env_cls)

    populate_dataset(args.episode, env_cls, dataset)


if __name__ == '__main__':
    main()