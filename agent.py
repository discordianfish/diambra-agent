#!/usr/bin/env python
import logging
import argparse
import diambra.arena

from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

STEPS = 100_000_000
N_PER_STATUS = 1_000
N_PER_CHECKPOINT = 10_000

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        prog='my agent',
        description='My agent competing in Diambra Arena.'
    )
    subparsers = parser.add_subparsers(help='operations')

    parser_train = subparsers.add_parser('train', help='Train a model to play Diambra Arena.')
    parser_train.add_argument('--log-dir', type=str, default='logs')
    parser_train.set_defaults(func=train)

    parser_play = subparsers.add_parser('play', help='Play a model to play Diambra Arena.')
    parser_play.add_argument('--agent-path', type=str, required=True)
    parser_play.set_defaults(func=play)


    args = parser.parse_args()
    args.func(args)

def train(args):
    env, num_envs = make_sb3_env(
        "doapp",
        {
            "hardcore": True,
            "frame_shape": [128, 128, 1],
        },
        {
            "reward_normalization": True,
            "frame_stack": 5,
        },
    )
    logger.info("Running %d environments", num_envs)

    agent = PPO('CnnPolicy', env, verbose=1)
    logger.info("Agent policy: %s", agent.policy)

    agent.learn(total_timesteps=STEPS, callback=CheckpointCallback(N_PER_CHECKPOINT, args.log_dir))

    env.close()

def play(args):
    agent = PPO.load(args.agent_path)
    env, num_envs = make_sb3_env(
        "doapp",
        {
            "hardcore": True,
            "frame_shape": [128, 128, 1],
        },
        {
            "reward_normalization": True,
            "frame_stack": 5,
        },
    )

    obs = env.reset()
    cumulative_reward = 0.0
    i = 0
    while True:
        i = i + 1
        env.render()

        action, _state = agent.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        cumulative_reward += reward

        if i % N_PER_STATUS == 0:
            reward_str = ""
            if any(x != 0 for x in cumulative_reward):
                reward_str = cumulative_reward
            logger.info("%d. rewards: %s, info %s", i, reward_str, info)

        if done.any():
            obs = env.reset()
            break

    env.close()

if __name__ == "__main__":
    main()
