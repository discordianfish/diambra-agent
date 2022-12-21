import logging

from stable_baselines3.common.callbacks import CheckpointCallback
import optuna

from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env
from stable_baselines3 import PPO

from .callbacks import TrialEvalCallback

logger = logging.getLogger(__name__)

def sample_params(trial: optuna.Trial):
    return {
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.001, log=True),
        "discount_factor": trial.suggest_float("discount_factor", 0.8, 0.99, log=True),
        "minibatch_size": trial.suggest_int("minibatch_size", 32, 256),
        "epochs": trial.suggest_int("epochs", 5, 10),
        "entropy_coef": trial.suggest_float("entropy_coef", 0.001, 0.01, log=True),
        "value_coef": trial.suggest_float("value_coef", 0.5, 1.0, log=True),
        "num_hidden_units": trial.suggest_int("num_hidden_units", 16, 1024),
        "activation_fn": trial.suggest_categorical("activation_fn", ["sigmoid", "tanh", "relu"]),
    }

class Agent:
    def __init__(self, game, settings, n_timesteps=int(2e4)):
        self.game = game
        self.settings = settings
        self.n_timesteps = n_timesteps

    def objective(self, trial: optuna.Trial) -> float:
        params = sample_params(trial)
        env, num_envs = make_sb3_env(
            self.game,
            self.settings,
            {
                "reward_normalization": True,
                "frame_stack": 5,
            },
        )
        logger.info("Running %d environments", num_envs)
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            gamma=params["gamma"],
            learning_rate=params["learning_rate"],
            discount_factor=params["discount_factor"],
            minibatch_size=params["minibatch_size"],
            epochs=params["epochs"],
            entropy_coef=params["entropy_coef"],
            value_coef=params["value_coef"],
            num_hidden_units=params["num_hidden_units"],
            activation_fn=params["activation_fn"],
        )
        eval_callback = TrialEvalCallback(env, trial, eval_freq=10000, verbose=1)

        nan = False
        try:
            model.learn(
                total_timesteps=self.n_timesteps,
                callback=eval_callback,
            )
        except AssertionError as e:
            logger.error("Assertion error: %s", e)
            nan = True
        finally:
            model.env.close()
            env.close()

        if nan:
            return float("nan")

        if eval_callback.is_pruned:
            raise optuna.exceptions.TrialPruned()
        return eval_callback.last_mean_reward

    def optimize(self, n_trials=100):
        study = optuna.create_study(direction="maximize")
        try:
            study.optimize(self.objective, n_trials=n_trials)
        except KeyboardInterrupt:
            pass

        return study.best_params


    def train(self, args):
        """train subcommand"""
        logger.info("Settings: %s", self.settings)
        env, num_envs = make_sb3_env(
            self.game,
            self.settings,
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
            total_timesteps=args.steps,
            callback=CheckpointCallback(
                args.n_per_checkpoint, args.log_dir, name_prefix=args.name_prefix)
        )

        env.close()

    def play(self, args):
        """play subcommand"""
        logger.info("Settings: %s", self.settings)
        agent = PPO.load(args.agent_path[0])
        env, _ = make_sb3_env(
            self.game,
            self.settings,
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

            if i % args.n_per_status == 0:
                logger.info("%d. rewards: %s, info %s", i, cumulative_reward, info)

            if done:
                obs = env.reset()
                break

        env.close()

