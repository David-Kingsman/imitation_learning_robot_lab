from typing import Type
from pathlib import Path
import argparse
import h5py
import numpy as np
import dataclasses
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from imitation_learning_lerobot.envs import Env, EnvFactory

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None

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
        video_backend=config.video_backend
    )

    return dataset

# populate dataset
def populate_dataset(env_cls: Type[Env], dataset: LeRobotDataset):
    task = env_cls.name

    h5_dir = Path(__file__).parent.parent.parent / Path("outputs/datasets") / Path(env_cls.name + "_hdf5")

    for item in h5_dir.iterdir():
        if not item.is_file():
            continue

        with h5py.File(item, 'r') as root:

            episode_length = root['/actions'].shape[0]

            for j in range(episode_length):
                frame = {
                    'observation.state': root['/observations/agent_pos'][j],
                    **{f'observation.images.{camera}': root[f'/observations/pixels/{camera}'][j] for camera in
                       env_cls.cameras},
                    'action': root['/actions'][j]
                }

                dataset.add_frame(frame, task=task)
            dataset.save_episode()


if __name__ == '__main__':
    args = parse_args()

    env_cls = EnvFactory.get_strategies(args.env_type)

    dataset = create_empty_dataset(env_cls)

    populate_dataset(env_cls, dataset)
