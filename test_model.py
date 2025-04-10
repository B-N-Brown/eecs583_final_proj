import gym
import compiler_gym
import numpy as np
from compiler_gym.wrappers import RewardWrapper

from runtime_reward import *


# creates a new environment (same as gym.make)
# selects the compiler to use
# selects the program to compile
# selects the observation space
# selects the optimization target

print(compiler_gym.COMPILER_GYM_ENVS)


env = compiler_gym.make(
    "llvm-v0",
    benchmark="cbench-v1/qsort", # the data set for HLS: benchmark://chstone-v0
    observation_space="Autophase",
    reward_space="IrInstructionCountOz"
)

# line added to implement custom reward
env = RuntimeImprovementWrapper(env)

# print("ACTION SPACE: ", env.observation.spaces["Autophase"].space)

action_spaces = [ env.action_space["-loop-unroll"],  env.action_space["-loop-reroll"]]
print(env.action_space["-loop-unroll"].space)

# starts a new compilation session
observation = env.reset()
# print("observation: ", observation)
# print()
# prints the IR of the program
#env.render()

MAX_EPOCHS = 2 
DISCOUNT_FACTOR = 0.99
N_TRIALS = 25
REWARD_THRESHOLD = 10 
PRINT_INTERVAL = 10
INPUT_DIM = env.observation_space.shape[0]
HIDDEN_DIM = 128
OUTPUT_DIM = env.action_space.n
DROPOUT = 0.5
LEARNING_RATE = 0.01
for episode in range(1, MAX_EPOCHS+1):
    observation, reward, done, info = env.step(action_spaces[0])
    mean_episode_return = np.mean(reward)
    print(info)
    # env.render()
    if episode % PRINT_INTERVAL == 0:
        print(episode, "Reward:", mean_episode_return)
    if mean_episode_return >= REWARD_THRESHOLD:
        print(f'Reached reward threshold in {episode} episodes')
        break

# # applies a random optimization, updates state/reward/actions
# print("ACTION SPACE:", env.action_space.sample())
# observation, reward, done, info = env.step(action_spaces[0])
# # this is what will run a custom action space: env.action_space["loop-unroll"]
# print("observation: ", observation)
# # print("reward: ", env.convert_reward(reward))
# print("reward: ", reward)
# print("done: ", done)
# print("info: ", info)

# print()

# observation, reward, done, info = env.step(env.action_space["-loop-unroll -unroll-count=2"])

# print("observation: ", observation)
# print("reward: ", reward)
# print("done: ", done)
# print("info: ", info)

# print()


# observation, reward, done, info = env.step(env.action_space["-loop-unroll"])
# # this is what will run a custom action space: env.action_space["loop-unroll"]
# print("observation: ", observation)
# # print("reward: ", env.convert_reward(reward))
# print("reward: ", reward)
# print("done: ", done)
# print("info: ", info)

# print()

# # closes the environment, freeing resources
# env.close()
