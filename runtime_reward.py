from compiler_gym.wrappers import RewardWrapper

class RuntimeImprovementWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_runtime = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_runtime = self.env.observation["Runtime"]
        return obs

    def convert_reward(self, reward):  # required by CompilerGym's RewardWrapper
        current_runtime = self.env.observation["Runtime"]
        if self.last_runtime is None:
            return 0.0
        reward = self.last_runtime - current_runtime  # reward improvement
        self.last_runtime = current_runtime
        return reward
