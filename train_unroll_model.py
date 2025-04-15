from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from loop_unroll_env import LoopUnrollEnv

gym_env = make_vec_env(LoopUnrollCompilerGymEnv, n_envs=4)
model = PPO("MlpPolicy", gym_env, verbose=1)
model.learn(total_timesteps=20000)