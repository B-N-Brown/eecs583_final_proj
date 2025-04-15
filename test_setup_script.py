import compiler_gym
import compiler_gym.envs

import logging
logging.basicConfig(level=logging.DEBUG)


env = compiler_gym.make("llvm-v0")
env.reset()