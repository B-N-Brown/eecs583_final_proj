import gym
from runtime_reward import RuntimeImprovementWrapper

base_env = gym.make("llvm-v0")
base_env.observation_space = "Runtime"

env = RuntimeImprovementWrapper(base_env)

obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print("Reward:", reward)
