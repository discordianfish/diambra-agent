#!/usr/bin/env python
"""Diambra Arena agent"""
import logging
import argparse
import time

from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

STEPS = 100_000_000
N_PER_STATUS = 100
N_PER_CHECKPOINT = 10_000

logger = logging.getLogger(__name__)
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


def main():
    """main entry point"""
    parser = argparse.ArgumentParser(
        prog='my agent',
        description='My agent competing in Diambra Arena.'
    )
    parser.add_argument('--difficulty', type=int, help='difficulty level')
    parser.add_argument('--characters', type=str, help='character name')

    subparsers = parser.add_subparsers(help='operations')

    parser_train = subparsers.add_parser(
        'train', help='Train a model to play Diambra Arena.')
    parser_train.add_argument('--log-dir', type=str, default='logs')
    parser_train.add_argument(
        '--name-prefix', type=str, default=time.strftime("%Y%m%d-%H%M%S"))
    parser_train.add_argument('--load-agent-path', type=str)
    parser_train.set_defaults(func=train)

    parser_play = subparsers.add_parser(
        'play', help='Play a model to play Diambra Arena.')
    parser_play.add_argument('agent_path', nargs=1, type=str)
    parser_play.set_defaults(func=play)

    args = parser.parse_args()
    args.func(args)


def settings_from_args(args):
    """create settings from argpase args"""
    settings = {
        "hardcore": True,
        "frame_shape": [128, 128, 1],
    }
    if args.difficulty:
        settings["difficulty"] = args.difficulty
    if args.characters:
        settings["characters"] = args.characters
    return settings


def train(args):
    """train subcommand"""
    env, num_envs = make_sb3_env(
        "doapp",
        settings_from_args(args),
        {
            "reward_normalization": True,
            "frame_stack": 5,
        },
    )
    logger.info("Running %d environments", num_envs)

    if args.load_agent_path:
        logger.info("Loading agent from %s", args.load_agent_path)
        agent = PPO.load(args.load_agent_path, env=env)
    else:
        agent = PPO('CnnPolicy', env, verbose=1)

    logger.info("Agent policy: %s", agent.policy)

    agent.learn(
        total_timesteps=STEPS,
        callback=CheckpointCallback(
            N_PER_CHECKPOINT, args.log_dir, name_prefix=args.name_prefix)
    )

    env.close()


def play(args):
    """play subcommand"""
    agent = PPO.load(args.agent_path[0])
    env, _ = make_sb3_env(
        "doapp",
        settings_from_args(args),
        {
            "reward_normalization": True,
            "frame_stack": 5,
        },
        no_vec=True,
    )

    logger.info("resetting env")
    obs = env.reset()
    logger.info("env resetted")
    cumulative_reward = 0.0
    i = 0
    while True:
        i = i + 1
        env.render()

        action, _state = agent.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        cumulative_reward += reward

        if i % N_PER_STATUS == 0:
            logger.info("%d. rewards: %s, info %s", i, cumulative_reward, info)

        if done:
            obs = env.reset()
            break

    env.close()


if __name__ == "__main__":
    main()
