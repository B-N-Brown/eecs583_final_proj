# Simple policy training script and makeshift environment sampling loop
import compiler_gym
from tqdm import tqdm
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from runtime_reward import RuntimeImprovementWrapper


def gym_training(env):
    """Simple loop to run through the GYM environment for sanity checking"""

    # GYM style training loop
    n_episodes = 1000 
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

def ppo_training(env):
    # Create vectorized env (recommended for SB3)
    vec_env = make_vec_env(lambda: env, n_envs=1)

    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=10000)


def a2c_training(env):
    # Wrap in a vectorized env (required by SB3)
    vec_env = make_vec_env(lambda: env, n_envs=1)

    # Optional: custom network
    policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])

    # Train
    model = A2C("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=50000)

    # Save or evaluate
    model.save("a2c_compilergym")


def main():
    # TODO: add scripts for loading model hyperparams

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
    seed = 42
    env.action_space.seed(seed)
    env.seed = seed 

    # print(env.seed())

    # action_spaces = [ env.action_space["-loop-unroll"],  env.action_space["-loop-reroll"]]
    # print("ACTION SPACE:", env.action_space)
    # print("OBSERVATION SPACE:", env.observation_space)
    # print(env.action_space.from_string("-loop-unroll"))

    # Train model on MLP policy network
    # ppo_training(env)    
    a2c_training(env)







if __name__ == "__main__":
    main()

