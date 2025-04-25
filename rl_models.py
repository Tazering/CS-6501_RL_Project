import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import mani_skill.envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv
import torch
import numpy as np

def run_ppo(env, total_timesteps, policy = "MlpPolicy", learning_rate = 3e-4, model_name = ""):
    ppo_model = PPO(policy = policy, env = env, learning_rate = learning_rate, n_steps = 2048, batch_size = 64, n_epochs = 10, gamma = 0.99, gae_lambda = .95,
    clip_range = .2, verbose = 1, seed = SEED)
    ppo_model.learn(total_timesteps = total_timesteps, progress_bar = True)
    ppo_model.save(model_name)

run_ppo()