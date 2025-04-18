import gym
import torch
import compiler_gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from compiler_gym.wrappers import RewardWrapper

from runtime_reward import RuntimeImprovementWrapper

import stable_baselines3
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from loop_actions import loop_opt_actions, loop_action_space


def test_model(env, checkpoint_name="basic_model.pth"):
    # model = model.load(path=f"model_checkpoints/{checkpoint_name}")

    model = PPO.load(f"model_checkpoints/{checkpoint_name}", 
                     print_system_info=True)
    
    averaged_reward = evaluate_policy(
                            env=env, 
                            model=model,
                            n_eval_episodes=1,
                            reward_threshold=0.0000001,
                            return_episode_rewards=True,              
                     ) 

    print("Averaged reward:", averaged_reward)

def test_model_loop(env, checkpoint_name="basic_model.pth"):
    # benchmarks = env.datasets["cbench-v1/sha"].benchmarks
    total_rewards = []
    prediction_times = []

    model = PPO.load(f"model_checkpoints/{checkpoint_name}", 
                     print_system_info=True)


    # for benchmark in benchmarks[:10]:
    #     obs = env.reset(benchmark=benchmark)
    #     done = False
    #     total_reward = 0

    #     while not done:
    #         action, _ = model.predict(obs, deterministic=True)
    #         obs, reward, done, info = env.step(action)
    #         total_reward += reward

    #     results.append((benchmark, total_reward, info))


    n_episodes = 1 
    # env = gym.wrappers.RecordEpisodeStatistics(env, n_episodes)
    done = False

    for episode in tqdm(range(n_episodes)):
        env.reset()
        total_reward = 0
        observation = env.reset()

        with tqdm(desc="Processing") as pbar:
            tqdm_counter = 0

            while not done:
                action, _ = model.predict(observation, deterministic=True)
                print("ACTION:", type(action))
                action = env.action_space.sample()
                print("ACTION1:", type(action))

                observation, reward, done, info = env.step(action)

                print("SAMPLD ACTION:", env.action_space.to_string(action))
                print("info: ", info)


                print("ACTION:", action)

                total_reward += reward
                
                tqdm_counter += 1
                pbar.update(1)
            
            # exit()

        print("Time to run inference on episode:", env.episode_walltime)
        total_rewards.append(total_reward)
        prediction_times.append(env.episode_walltime)

    plt.plot(total_reward)
    plt.savefig("test_plot.png")

def main():
    env = compiler_gym.make(
        "llvm-v0",
        benchmark="cbench-v1/sha", 
        observation_space="Autophase",
        reward_space="IrInstructionCountOz"
    )
    env = RuntimeImprovementWrapper(env)
    env.runtime_observation_count = 1

    env.action_space = loop_action_space

    env = RuntimeImprovementWrapper(env)
    env.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("USING DEVICE:", device)

    # Testing code
    test_model_loop(env)




if __name__ == "__main__":
    main()
