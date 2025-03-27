from compiler_gym.envs.llvm import LlvmEnv, LlvmReward

class RuntimeImprovementReward(LlvmReward):
    def __init__(self):
        super().__init__()
        self.last_runtime = None

    def reset(self, env: LlvmEnv):
        # Initialize or reset state when the env resets
        self.last_runtime = env.observation["runtime"]

    def __call__(self, env: LlvmEnv) -> float:
        current_runtime = env.observation["runtime"]
        if self.last_runtime is None:
            return 0.0
        reward = self.last_runtime - current_runtime
        self.last_runtime = current_runtime
        return reward

    def close(self):
        pass
