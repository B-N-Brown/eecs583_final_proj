import gym
import compiler_gym
import numpy as np

# creates a new environment (same as gym.make)
# selects the compiler to use
# selects the program to compile
# selects the observation space
# selects the optimization target
env = compiler_gym.make(
    "llvm-v0",
    benchmark="cbench-v1/qsort", # the data set for HLS: benchmark://chstone-v0
    observation_space="Autophase",
    reward_space="IrInstructionCountOz",
)
# starts a new compilation session
env.reset()
# prints the IR of the program
env.render()
# applies a random optimization, updates state/reward/actions
env.step(env.action_space.sample())
# this is what will run a custom action space: env.action_space["loop-unroll"]

# closes the environment, freeing resources
env.close()