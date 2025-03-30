<<<<<<< HEAD
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
=======
from compiler_gym.spaces import Reward

class RuntimeReward(Reward):
    """An example reward that uses changes in the "runtime" observation value
    to compute incremental reward.
    """

    baseline_runtime: int

    def __init__(self):
        super().__init__(
            name="runtime",
            observation_spaces=["runtime"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.baseline_runtime = 0

    def reset(self, benchmark: str, observation_view):
        del benchmark  # unused
        self.baseline_runtime = observation_view["runtime"]

    def update(self, action, observations, observation_view):
        del action  # unused
        del observation_view  # unused
        return float(self.baseline_runtime - observations[0]) / self.baseline_runtime
>>>>>>> 8f8e6a3c06354f42f4985215a1f127ee2dafe719
