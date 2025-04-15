import compiler_gym
import gym
from compiler_gym.spaces import Discrete, Box
import numpy as np

class LoopUnrollEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = compiler_gym.make("llvm-v0",observation_space="autophase")
        self.env.reset()
        # Unroll factors to choose from
        self.unroll_factors = [2, 4, 8, 16]
        self.action_space = Discrete(len(self.unroll_factors), "unroll options")
    def reset(self):
        self.env.reset()
        return self.env.observation["autophase"]
    def step(self, action):
        unroll = self.unroll_factors[action]
        passes = [
            "-loop-unroll",
            f"-unroll-count={unroll}",
            "-unroll-allow-partial",
            "-simplifycfg",
        ]
        try:
            self.env.write_ir(passes=passes)
            reward = -self.env.observation["runtime"]
        except Exception:
            # Fallback if LLVM errors
            reward = -1e6
        obs = self.env.observation["autophase"]
        done = True
        return obs, reward, done, {"unroll_factor": unroll}