import compiler_gym
from runtime_reward import RuntimeImprovementWrapper

env = compiler_gym.make("llvm-v0")
env = RuntimeImprovementWrapper(env)

obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward}")