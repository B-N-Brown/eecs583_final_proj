import numpy as np
from compiler_gym.wrappers import RewardWrapper

class RuntimeImprovementWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_runtime = 0.0
        self.env.runtime_observation_count = 2

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Get the initial runtime
        self.last_runtime = self.env.observation["Runtime"]
        return obs
    
    def step(self, action, observation=None):
        # action = int(action)
        observation, reward, done, info = self.env.step(action)
        reward = self.convert_reward(reward)
        return observation, reward, done, info 

    def convert_reward(self, reward):  # required method
        # Get current runtime
        current_runtime = self.env.observation["Runtime"]
        if self.last_runtime is None or current_runtime is None:
            return 0.0
        reward = self.last_runtime - current_runtime
        self.last_runtime = current_runtime

        # For compatibility with stable baselines
        # convert to np array and average runtime observation count
        # TODO: put this before the array subtraction
        reward = np.mean(np.array(reward))

        return reward
