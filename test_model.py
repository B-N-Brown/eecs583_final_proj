import gym
import compiler_gym
import numpy as np
from tqdm import tqdm
from compiler_gym.wrappers import RewardWrapper

from runtime_reward import RuntimeImprovementWrapper



def test_model(env, checkpoint):
    # model = model.load(path=f"model_checkpoints/{checkpoint_name}")
    return


def main():
    env = compiler_gym.make(
        "llvm-v0",
        benchmark="cbench-v1/sha", 
        observation_space="Autophase",
        reward_space="IrInstructionCountOz"
    )
    env = RuntimeImprovementWrapper(env)
    env.runtime_observation_count = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("USING DEVICE:", device)

    # Testing code
    ppo_training_sb(env, device)




if __name__ == "__main__":
    main()
