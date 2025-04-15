import compiler_gym
import gym
from compiler_gym.spaces import Discrete, Box
import numpy as np

class LoopUnrollEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = compiler_gym.make("llvm-v0", observation_space="Autophase")
        self.env.reset()
        # Unroll factors to choose from
        self.unroll_factors = [2, 4, 8, 16]
        self.action_space = Discrete(len(self.unroll_factors), "unroll options")

        self.observation_space = self.env.observation_space

    def reset(self):
        self.env.reset()
        return self.env.observation["Autophase"]
    def step(self, action):
        unroll = self.unroll_factors[action]
        passes = [
            "-loop-unroll",
            f"-unroll-count={unroll}",
            "-unroll-allow-partial",
            "-simplifycfg",
        ]
        try:
            #self.env.write_ir(passes=passes)
            self.env.step(passes)
            reward = -self.env.observation["Runtime"]
        except Exception as e:
            print(e)
            # Fallback if LLVM errors
            reward = -1e6
        obs = self.env.observation["Autophase"]
        done = True
        return obs, reward, done, {"unroll_factor": unroll}