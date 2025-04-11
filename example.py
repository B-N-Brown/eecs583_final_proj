# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import compiler_gym
from compiler_gym.spaces import NamedDiscrete
from runtime_reward import RuntimeImprovementWrapper


env = compiler_gym.make(
    "llvm-ic-v0",
    benchmark="cbench-v1/jpeg-d",
    observation_space="Autophase"
)

env.action_space.seed(21)
    
# Ex dataset
#obs_feat_vec = env.observation_space[3]
#print ("Total instruction count: ", obs_feat_vec[51])

# actions =["-loop-unroll -unroll-count=2 -unroll-force",
#         "-loop-unroll -unroll-count=4 -unroll-force",
#         "-loop-unroll -unroll-count=8 -unroll-force"]

#actions = ["-die"]
#actions = ["-loop-unroll"]

# ["-loop-unroll", "-unroll-count=4", "-unroll-force"]

# action_names = ["force-unroll-4"]



# custom_as = NamedDiscrete(actions)
# env.action_space = custom_as
env = RuntimeImprovementWrapper(env)


print("SAMPLED ACTION SPACE:", env.action_space.sample())


observation = env.reset()
print("observation: ", observation)

print(f"AUTOPHASE DICTIONARY 0: {env.observation['AutophaseDict']['TotalInsts']}")

print()

for i in range(10):
    sampled_action = env.action_space["-loop-unroll -unroll-count=2"]
    print("SAMPLD ACTION:", env.action_space.to_string(sampled_action))

    observation, reward, done, info = env.step(sampled_action)
    print("\n\nFUCK TOU")

    print("_______________________________________")
    print("observation: ", observation)
    print("reward: ", reward)
    print("done: ", done)
    print("info: ", info)
    print(f"AUTOPHASE TOTAL INSTR ROUND {i + 1}: {env.observation['AutophaseDict']['TotalInsts']}")
    print("CHAT HELP ME :", env.observation["IrInstructionCountOz"])
    print("_______________________________________")

# print()

# observation, reward, done, info = env.step(env.action_space.sample())
# print("observation: ", observation)
# print("reward: ", reward)
# print("done: ", done)
# print("info: ", info)

env.reset()

# TODO: implement write_bitcode(..) or write_ir(..)
# env.write_bitcode("/tmp/output.bc")