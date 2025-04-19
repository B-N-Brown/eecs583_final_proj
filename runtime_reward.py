import numpy as np
from compiler_gym.wrappers import RewardWrapper

class RuntimeImprovementWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env.last_runtime = 0.0
        self.env.runtime_observation_count = 2

    # Commented out for dset wrapper
    # TODO: check for any issues from commenting out
    # def reset(self, **kwargs):
    #     obs = self.env.reset(**kwargs)
    #     # Get the initial runtime
    #     self.env.last_runtime = np.mean(self.env.observation["Runtime"])
    #     return obs
    
    def step(self, action, observation=None):
        # action = int(action)
        observation, reward, done, info = self.env.step(action)
        reward = self.convert_reward(reward)
        # print("reward ratio: ", self.reward_ratio)
        done = self.env.reward_ratio < 0.002
        return observation, reward, done, info 

    def convert_reward(self, reward):  # required method
        # Get current runtime
        current_runtime = self.env.observation["Runtime"]
        current_runtime = np.mean(np.array(current_runtime))

        if self.env.last_runtime is None or current_runtime is None:
            return 0.0

        reward = self.env.last_runtime - current_runtime
        self.env.reward_ratio = abs(reward)/self.env.last_runtime
        self.env.last_runtime = current_runtime

        return reward


class CodesizeNRuntimeImprovementWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env.last_runtime = 0.0
        self.env.runtime_observation_count = 2

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Get the initial runtime
        self.env.last_runtime = np.mean(self.env.observation["Runtime"])
        return obs
    
    def step(self, action, observation=None):
        # action = int(action)
        observation, reward, done, info = self.env.step(action)
        reward = self.convert_reward(reward)
        # print("reward ratio: ", self.reward_ratio)
        done = self.env.reward_ratio < 0.001
        return observation, reward, done, info 

    def convert_reward(self, reward):  # required method
        # Get current runtime
        current_runtime = self.env.observation["Runtime"]
        current_runtime = np.mean(np.array(current_runtime))

        if self.env.last_runtime is None or current_runtime is None:
            return 0.0

        reward = self.env.last_runtime - current_runtime
        self.env.reward_ratio = abs(reward)/self.env.last_runtime
        self.env.last_runtime = current_runtime

        # TODO Get code size




        # TODO: Balance loss

        
        return reward
