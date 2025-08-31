from typing import Type
from pathlib import Path
import argparse

from loop_rate_limiters import RateLimiter
import numpy as np
import h5py
import cv2

from imitation_learning_robot_lab.envs import Env, EnvFactory
from imitation_learning_robot_lab.teleoperation import HandlerFactory


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
        '--handler.type',
        type=str,
        dest='handler_type',
        required=True,
        help='handler type'
    )

    return parser.parse_args()

# teleoperate the robot with the teleoperator
def teleoperate(env_cls: Type[Env], handler_type):
    handler_cls = HandlerFactory.get_strategies(env_cls.name + "_" + handler_type)
    handler = handler_cls()
    handler.start()
    handler.print_info()

    env = env_cls(render_mode="human")
    observation, info = env.reset()

    # GUI window
    for camera in env_cls.cameras:
        cv2.namedWindow(camera, cv2.WINDOW_GUI_NORMAL)

    data_dict = {
        '/observations/agent_pos': [],
        **{f'/observations/pixels/{camera}': [] for camera in env_cls.cameras},
        '/actions': []
    }

    rate_limiter = RateLimiter(frequency=env.control_hz)

    action = handler.action
    last_action = action.copy()
    while not handler.done:
        if not handler.sync:
            rate_limiter.sleep()
            continue

        last_action[:] = action
        action[:] = handler.action
        if np.max(np.abs(action - last_action)) > 1e-6:
            data_dict['/observations/agent_pos'].append(observation['agent_pos'])
            for camera in env_cls.cameras:
                data_dict[f'/observations/pixels/{camera}'].append(observation['pixels'][camera])
            data_dict['/actions'].append(action)
        else:
            action[:] = last_action

        observation, reward, terminated, truncated, info = env.step(action)

        env.render()
        # image display by opencv for each camera
        for camera in env.cameras:
            cv2.imshow(camera, cv2.cvtColor(observation["pixels"][camera], cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        rate_limiter.sleep()

    cv2.destroyAllWindows()
    handler.close()
    env.close()

    return data_dict


def write_to_h5(env_cls: Type[Env], data_dict: dict):
    h5_dir = Path(__file__).parent.parent.parent / Path("outputs/datasets") / Path(env_cls.name + "_hdf5")
    h5_dir.mkdir(parents=True, exist_ok=True)

    index = len([f for f in h5_dir.iterdir() if f.is_file()])

    h5_path = h5_dir / Path(f"episode_{index:06d}.hdf5")

    with h5py.File(h5_path, 'w', ) as root:

        episode_length = len(data_dict['/actions'])

        obs = root.create_group('observations')

        obs.create_dataset('agent_pos', (episode_length, env_cls.state_dim), dtype='float32', compression='gzip')

        pixels = obs.create_group('pixels')
        for camera in env_cls.cameras:
            shape = (episode_length, env_cls.height, env_cls.width, 3)
            chunks = (1, env_cls.height, env_cls.width, 3)
            pixels.create_dataset(camera, shape=shape, dtype='uint8', chunks=chunks, compression='gzip')

        root.create_dataset('actions', (episode_length, env_cls.action_dim), dtype='float32', compression='gzip')

        for name, array in data_dict.items():
            root[name][...] = array


def main():
    args = parse_args()

    env_cls = EnvFactory.get_strategies(args.env_type)

    data_dict = teleoperate(env_cls, args.handler_type)

    write_to_h5(env_cls, data_dict)


if __name__ == '__main__':
    main()
