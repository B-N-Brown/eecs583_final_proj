# Simple policy training script and makeshift environment sampling loop
import torch
import compiler_gym
from compiler_gym.wrappers.commandline import ConstrainedCommandline
from tqdm import tqdm

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env

from loop_actions import loop_opt_actions, loop_action_space

from runtime_reward import RuntimeImprovementWrapper

# STABLE BASELINES TRAINING REGIMES
def ppo_training_sb(env, device, checkpoint_name="basic_model.pth"):
    n_steps = 256
    n_envs = 8
    total_timesteps = n_steps * n_envs # PPO training min

    # Create vectorized env (recommended for SB3)
    vec_env = make_vec_env(lambda: env, n_envs=n_envs)
    
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1, 
        batch_size=256,
        n_steps=n_steps,
        n_epochs=10,
        seed=42,
        device=device
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(path=f"model_checkpoints/{checkpoint_name}")


def a2c_training_sb(env):
    total_timesteps = 2048

    # Wrap in a vectorized env (required by SB3)
    vec_env = make_vec_env(lambda: env, n_envs=1)

    # Optional: custom network
    policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])

    # Train
    model = A2C("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # Save or evaluate
    model.save("a2c_compilergym")


def main():
    # TODO: add scripts for loading model hyperparams

    # Checking out the dataset
    # compiler_gym.envs.llvm.datasets.CBenchDataset("cbench-v1/sha")
    print("PRINTING:", compiler_gym.envs.llvm.datasets.get_llvm_datasets("anghabench-v1"))

    # Wrap the environment to restrict action space
    env = compiler_gym.make(
        "llvm-v0",
        benchmark="cbench-v1/sha", 
        observation_space="Autophase",
        reward_space="IrInstructionCountOz"
    )


    # Restrict action space to loop actions
    env.action_space = loop_action_space
    # env = ConstrainedCommandline(env, loop_opt_actions)

    # Incorporate Custom Runtime Reward
    env = RuntimeImprovementWrapper(env)
    env.reset()

    seed = 42

    env.action_space.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("USING DEVICE:", device)

    # Train model on MLP policy network
    # basic_train(env)
    ppo_training_sb(env, device)


if __name__ == "__main__":
    main()

