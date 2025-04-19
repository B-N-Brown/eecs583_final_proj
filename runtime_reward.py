import numpy as np
from compiler_gym.wrappers import RewardWrapper

class RuntimeImprovementWrapper(RewardWrapper):
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

        return reward


class CodesizeNRuntimeImprovementWrapper(RewardWrapper):
    def __init__(self, env, alpha):
        super().__init__(env)
        self.env.last_runtime = 0.0 
        self.env.runtime_observation_count = 2
        self.alpha = alpha

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Get the initial runtime
        self.env.last_runtime = np.mean(self.env.observation["Runtime"])
        self.env.last_reward = self.env.last_runtime # initialize the reward to the runtime
        return obs
    
    def step(self, action, observation=None):
        # action = int(action)
        observation, code_size_reward, done, info = self.env.step(action) ## only works if env intialized w/ ir instruction count
        total_reward = self.convert_reward(code_size_reward)
        # print("reward ratio: ", self.reward_ratio)
        done = self.env.reward_ratio < 0.001
        return observation, total_reward, done, info 

    def convert_reward(self, code_size_reward):  # required method
        # Get current runtime
        current_runtime = self.env.observation["Runtime"]
        current_runtime = np.mean(np.array(current_runtime))

        if self.env.last_runtime is None or current_runtime is None:
            return 0.0

        runtime_reward = self.env.last_runtime - current_runtime

        total_reward = self.alpha*runtime_reward + (1-self.alpha)*code_size_reward

        self.env.reward_ratio = abs(total_reward - self.last_reward)/self.last_reward
        self.env.last_reward = total_reward

        return total_reward
