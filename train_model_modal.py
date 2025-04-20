# Simple policy training script and makeshift environment sampling loop
import modal # Yall will probably have to comment this out

image = (
    # 1) Use an officially supported CUDA image
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .add_local_dir(local_path="/home/bnb/Documents/uofm/eecs583/final_proj",
                   remote_path="/final_proj",
                   ignore=["env", ".git", ".devcontainer", "examples"],
                   copy=True
    )
    .pip_install(["setuptools==65.5.0", "pip==21", "wheel==0.38.0"])
    .pip_install_from_requirements("requirements.txt")
)

import gym
import torch
import compiler_gym
from tqdm import tqdm
from pprint import pprint
from torch import optim

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from eecs583_final_proj.runtime_reward import RuntimeInstCountRewardWrapper

app = modal.App("583_proj_training", image=image)

# STABLE BASELINES TRAINING REGIMES
@app.function(gpu="A10G")
def ppo_training_sb(checkpoint_name="basic_model.pth"):
    
    # With modal you can't pass the env in
    env = compiler_gym.make(
        "llvm-v0",
        benchmark="cbench-v1/sha", 
        observation_space="Autophase",
        reward_space="IrInstructionCountOz"
    )
    # line added to implement custom reward
    env = RuntimeInstCountRewardWrapper(env)
    # seed = 42
    # env.action_space.seed(seed)
    env.runtime_observation_count = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("USING DEVICE:", device)


    # Create vectorized env (recommended for SB3)
    vec_env = make_vec_env(lambda: env, n_envs=8)

    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1, 
        batch_size=256,
        n_epochs=10,
        seed=42,
        device=device
    )
    model.learn(total_timesteps=2048, progress_bar=True)
    model.save(path=f"model_checkpoints/{checkpoint_name}")
    
    return


def a2c_training_sb(env):
    # Wrap in a vectorized env (required by SB3)
    vec_env = make_vec_env(lambda: env, n_envs=1)

    # Optional: custom network
    policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])

    # Train
    model = A2C("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=5000, progress_bar=True)

    # Save or evaluate
    model.save("a2c_compilergym")

# RLLib TRAINING REGIMES
def env_creator(*args, **kwargs):
    env = compiler_gym.make(
        "llvm-v0",
        benchmark="cbench-v1/sha", 
        observation_space="Autophase",
        reward_space="IrInstructionCountOz"
    )
    # line added to implement custom reward
    env = RuntimeInstCountRewardWrapper(env)
    # env = EnvCompatibility(env)
    # seed = 42
    # env.action_space.seed(seed)

    env.runtime_observation_count = 1
    return env

def ppo_training_rllib():
    register_env("compiler_gym_env", env_creator)

    env = env_creator()
    ray.rllib.utils.check_env(env)

    ray.init(ignore_reinit_error=True)

    # Configure the algorithm.
    config = (
        PPOConfig()
        .environment(
            "compiler_gym_env",
            disable_env_checking=True,
            env_config={}
        )
    )

    algo = config.build()
    for _ in range(10):
        pprint(algo.train())


    algo.stop()

# TRAINING THE LOOONG WAY
def select_action(policy_net, state):
    state = torch.FloatTensor(state).unsqueeze(0)
    probs = policy_net(state)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)

def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def basic_train(env, checkpoint_name="basic_model.pth", episodes=50, gamma=0.99, lr=1e-2):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("USING DEVICE:", device)

    policy_net = PolicyNetwork(state_dim, action_dim).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    for episode in tqdm(range(episodes)):
        state = env.reset()
        state = torch.from_numpy(state).to(device)

        log_probs = []
        rewards = []

        with tqdm(desc="Processing") as pbar:
            tqdm_counter = 0

            done = False
            while not done:
                action, log_prob = select_action(policy_net, state)
                next_state, reward, done, _ = env.step(action)

                log_probs.append(log_prob)
                rewards.append(reward)
                state = next_state

                tqdm_counter += 1
                pbar.update(tqdm_counter)

        returns = compute_returns(rewards, gamma)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = -torch.sum(torch.stack(log_probs) * returns)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        print(f"Episode {episode}, total reward: {total_reward}")

    torch.save(f"model_checkpoints/{checkpoint_name}")
    env.close()

def basic_eval(env, checkpoint_name="basic_model.pth"):
    return

@app.local_entrypoint()
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
    env = RuntimeInstCountRewardWrapper(env)
    # seed = 42
    # env.action_space.seed(seed)
    env.runtime_observation_count = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("USING DEVICE:", device)


    # env.seed = lambda self, x : self.action_space.seed(x)

    # print(env.seed())

    # action_spaces = [ env.action_space["-loop-unroll"],  env.action_space["-loop-reroll"]]
    # print("ACTION SPACE:", env.action_space)
    # print("OBSERVATION SPACE:", env.observation_space)
    # print(env.action_space.from_string("-loop-unroll"))

    # Train model on MLP policy network
    # basic_train(env)
    ppo_training_sb.remote()







if __name__ == "__main__":
    main()

