import numpy as np
import compiler_gym
from compiler_gym.spaces import Reward
from compiler_gym.wrappers import RewardWrapper
from compiler_gym.util.registration import register

from gym.envs.registration import register as gym_register


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
    
    def step(self, action):
        # print(f"action from step: {action}")
        action = int(action)
        observation, reward, done, info = self.env.step(action)
<<<<<<< HEAD
        return observation, self.convert_reward(reward), done, info 
=======
        return observation, self.reward(reward), done, info 
>>>>>>> c30dd17c2e4345d8b9dec41e5d44d8c44df4856b

    def reward(self, reward):  # required method
        # Get current runtime
        current_runtime = self.env.observation["Runtime"]
        if self.last_runtime is None or current_runtime is None:
            return 0.0
        reward = self.last_runtime - current_runtime
        self.last_runtime = current_runtime

        # print("CURR RUN:", current_runtime)
        # print(self.env.observation)
        
        # print("obs count:", self.env.runtime_observation_count)

        # For compatibility with stable baselines
        # convert to np array and average runtime observation count
        reward = np.mean(np.array(reward))

        return reward
    

# EXAMPLE WORK
# class RuntimeImprovementWrapper(Reward):
#     """An example reward that uses changes in the "runtime" observation value
#     to compute incremental reward.
#     """

#     def __init__(self):
#         super().__init__(
#             id="runtime",
#             observation_spaces=["runtime"],
#             default_value=0,
#             default_negates_returns=True,
#             deterministic=False,
#             platform_dependent=True,
#         )
#         self.baseline_runtime = 0

#     def reset(self, benchmark: str, observation_view):
#         del benchmark  # unused
#         self.baseline_runtime = observation_view["runtime"]

#     def update(self, action, observations, observation_view):
#         del action  # unused
#         del observation_view  # unused
#         return float(self.baseline_runtime - observations[0]) / self.baseline_runtime


# def register(id: str, **kwargs):
#     compiler_gym.COMPILER_GYM_ENVS.append(id)
#     gym_register(id=id, **kwargs)


# register(
#     id="unrolling-py-v0",
#     entry_point="compiler_gym.service.client_service_compiler_env:ClientServiceCompilerEnv",
#     kwargs={
#         # "service": UNROLLING_PY_SERVICE_BINARY,
#         "rewards": [RuntimeImprovementWrapper().id]#, SizeReward()],
#         # "datasets": [UnrollingDataset()],
#     },
# )
