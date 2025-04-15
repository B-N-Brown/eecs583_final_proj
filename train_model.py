import compiler_gym
from tqdm import tqdm

from runtime_reward import RuntimeImprovementWrapper

# Checking out the dataset
# compiler_gym.envs.llvm.datasets.CBenchDataset("cbench-v1/sha")
print("PRINTING:", compiler_gym.envs.llvm.datasets.get_llvm_datasets("anghabench-v1"))

# Initialize environment
env = compiler_gym.make(
    "llvm-v0",
    benchmark="cbench-v1/sha", 
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


# GYM style training loop
n_episodes = 10
env.reset()
done = False

print(f"AUTOPHASE DICTIONARY 0: {env.observation['AutophaseDict']['TotalInsts']}")
for episode in tqdm(range(n_episodes)):
    env.reset()

    with tqdm(desc="Processing") as pbar:
        tqdm_counter = 0
        while not done:
            action = env.action_space.sample()  # agent policy that uses the observation and info

            instr_count_before = env.observation['AutophaseDict']['TotalInsts']
            observation, reward, done, info = env.step(action)

            if env.action_space.to_string(action) == "-loop-unroll":
                break
            tqdm_counter += 1
            pbar.update(1)
        
        # exit()


env.close()

exit()

# starts a new compilation session
observation = env.reset()
