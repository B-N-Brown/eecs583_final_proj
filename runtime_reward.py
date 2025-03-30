from compiler_gym.wrappers import RewardWrapper

class RuntimeImprovementWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_runtime = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Get the initial runtime
        self.last_runtime = self.env.observation["Runtime"]
        return obs

    def convert_reward(self, reward):  # required method
        # Get current runtime
        current_runtime = self.env.observation["Runtime"]
        if self.last_runtime is None or current_runtime is None:
            return 0.0
        reward = self.last_runtime - current_runtime
        self.last_runtime = current_runtime
        return reward
