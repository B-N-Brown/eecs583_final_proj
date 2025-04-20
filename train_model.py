"""
Simple policy training script and makeshift environment sampling loop

TODO List:
    * Investigate different datasets --> seems like cbench isn't suited for loop unrolling?
    * Try different feature vectors. Autophase might be too big
    * Evaluate different learning frameworks:
        - A2C
        - DQN
        - Others?
    * Finetune ppo params (how tf does this work?)
"""
import torch
import compiler_gym
from compiler_gym.wrappers import CycleOverBenchmarks, IterateOverBenchmarks
from stable_baselines3.common.logger import configure

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env

from loop_actions import loop_action_space

from runtime_reward import RuntimeImprovementWrapper, CodesizeNRuntimeImprovementWrapper
from dataset_wrapper import CBenchWrapper

# STABLE BASELINES TRAINING REGIMES
def ppo_training_sb(env, device, checkpoint_name="basic_model.pth"):
    n_steps = 5
    n_envs = 2
    total_timesteps = n_steps * n_envs # PPO training min

    # Create vectorized env (recommended for SB3)
    vec_env = make_vec_env(lambda: env, n_envs=n_envs)
    
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=2, 
        batch_size=256,
        n_steps=n_steps,
        n_epochs=1,
        seed=42,
        device=device
    )
    log_path = 'logs/'
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    model.save(path=f"model_checkpoints/{checkpoint_name}")


def a2c_training_sb(env, checkpoint_name="basic_model.pth"):
    total_timesteps = 2048

    # Wrap in a vectorized env (required by SB3)
    vec_env = make_vec_env(lambda: env, n_envs=8)

    # Optional: custom network
    # policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])

    # Train
    model = A2C("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model.save(path=f"model_checkpoints/{checkpoint_name}")


def main():
    # Checking out the dataset
    # compiler_gym.envs.llvm.datasets.CBenchDataset("cbench-v1/sha")

    # Uncomment to install all datasets (takes a couple mins)
    # print("PRINTING:", compiler_gym.envs.llvm.datasets.get_llvm_datasets())
    # for dset in compiler_gym.envs.llvm.datasets.get_llvm_datasets():
    #     print(dset.install())

    angha_dset = compiler_gym.envs.llvm.datasets.AnghaBenchDataset("anghabench-v1")
    cbench_dset = compiler_gym.envs.llvm.datasets.CBenchDataset("cbench")
    print(angha_dset.benchmark_uris())

    # Wrap the environment to restrict action space
    env = compiler_gym.make(
        "llvm-v0",
        # benchmark="cbench/sha", 
        observation_space="Autophase",
        reward_space="IrInstructionCountOz"
    )

    print("dset size:", angha_dset.size)

    # Restrict action space to loop actions
    env.action_space = loop_action_space

    # Incorporate Custom Runtime Reward
    env = RuntimeImprovementWrapper(env)
    # env = CodesizeNRuntimeImprovementWrapper(env, alpha=0.5)

    env = CycleOverBenchmarks(env, angha_dset.benchmark_uris())
    env.reset()

    seed = 42

    env.action_space.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("USING DEVICE:", device)

    # Train model on MLP policy network
    # basic_train(env)

    ppo_training_sb(env, device, checkpoint_name="mlp_cbench_50epochs.pth")
    # a2c_training_sb(env, checkpoint_name="a2c_mlp_50epochs.pth")


if __name__ == "__main__":
    main()

