import compiler_gym
import gym
import random
import numpy as np

class CBenchWrapper(gym.Env):
    def __init__(self, env):
        super().__init__(env)
        # self.env = compiler_gym.make("llvm-v0", observation_space=observation_space, reward_space=reward_space)
        self.benchmarks = self.env.datasets["cbench"].benchmark_uris()
        
        # Use the underlying CompilerGym spaces
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        # Sample a new benchmark every time reset is called
        benchmark = random.choice(self.benchmarks)
        self.env.reset(benchmark=benchmark)
        return self.env.observation["Autophase"]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def render(self, mode="human"):
        return self.env.render()

    def close(self):
        self.env.close()
