import gym
import compiler_gym
import numpy as np
from tqdm import tqdm
from compiler_gym.wrappers import RewardWrapper

from runtime_reward import *


# creates a new environment (same as gym.make)
# selects the compiler to use
# selects the program to compile
# selects the observation space
# selects the optimization target

print(compiler_gym.COMPILER_GYM_ENVS)

# Checking out the dataset
compiler_gym.envs.llvm.datasets.CBenchDataset("cbench-v1/sha")

# Initialize environment
env = compiler_gym.make(
    "llvm-v0",
    benchmark="cbench-v1/sha", # the data set for HLS: benchmark://chstone-v0
    observation_space="Autophase",
    reward_space="IrInstructionCountOz"
)
# line added to implement custom reward
env = RuntimeImprovementWrapper(env)
env.action_space.seed(42)


action_spaces = [ env.action_space["-loop-unroll"],  env.action_space["-loop-reroll"]]
print("ACTION SPACE:", env.action_space)
print("OBSERVATION SPACE:", env.observation_space)


print(env.action_space.from_string("-loop-unroll"))

# exit()


# GYM style training loop
n_episodes = 5
# env = gym.wrappers.RecordEpisodeStatistics(env, n_episodes)
env.reset()
done = False

print(f"AUTOPHASE DICTIONARY 0: {env.observation['AutophaseDict']['TotalInsts']}")
for episode in tqdm(range(n_episodes)):
    env.reset()

    with tqdm(desc="Processing") as pbar:
        tqdm_counter = 0
        while not done:
            action = env.action_space.sample()  # agent policy that uses the observation and info
            action = env.action_space["-loop-unroll"]

            # print("SAMPLD ACTION:", env.action_space.to_string(action))
            observation, reward, done, info = env.step(action)
            # observation, reward, terminated, truncated, info = env.step(action)

            # done = terminated or truncated

            if not info['action_had_no_effect']:
                print("SUCCESS")

            # print("_______________________________________")
            # print("observation: ", observation)
            # print("reward: ", reward)
            # print("done: ", done)
            # print("info: ", info)
            # print(f"AUTOPHASE TOTAL INSTR ROUND : {env.observation['AutophaseDict']['TotalInsts']}")
            # print("CHAT HELP ME :", env.observation["IrInstructionCountOz"])
            # print("_______________________________________")

            tqdm_counter += 1
            pbar.update(1)
        
        # exit()


env.close()

exit()

# starts a new compilation session
observation = env.reset()
# print("observation: ", observation)
# print()
# prints the IR of the program
# env.render()

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
