import sys

sys.path.append('.')
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, EvalCallback

from CybORG.Agents.Wrappers.EnumActionWrapper import EnumActionWrapper
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
# import CybORG.Agents.Wrappers.ChallengeWrapper as cw
from CybORG.Agents.Wrappers.myChallengeWrapper import MyChallengeWrapper
from CybORG.Agents.Wrappers.ChallengeWrapper import ChallengeWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, GreenAgent, BlueMonitorAgent

CHECKPOINTS_MODELS = Path("checkpoints")
SAVE_FREQ = 10000000
TIMESTEPS = 300000000
STEP_SIZE = 3
N_ENVS = 64
CHECKPOINTS_PATH = "checkpoints"
MODELS_PATH = "models"
VIDEO_PATH = "videos"
RES_PATH = "res"
SCENARIO_PATH = "CybORG/Shared/Scenarios/Scenario1b.yaml"
lr = 0.0005


def get_cw(path: str):
    agents = {
        'Red': B_lineAgent,
        'Green': GreenAgent
    }
    cyborg = CybORG(path, 'sim', agents=agents)

    return ChallengeWrapper(env=cyborg, agent_name='Blue', max_steps=30)


def get_mcw(path: str) -> MyChallengeWrapper:
    agents = {
        'Red': B_lineAgent
        # 'Green': GreenAgent
    }

    env = CybORG(path, 'sim', agents=agents)
    eer = EnumActionWrapper(env)
    wrappers = FixedFlatWrapper(eer)
    # env = OpenAIGymWrapper(env=wrappers, agent_name='Blue')
    env = MyChallengeWrapper(env=wrappers, agent_name="Blue", max_counter=30)

    # env = table_wrapper(env, output_mode='vector')
    # env = EnumActionWrapper(env)
    # env = OpenAIGymWrapper(agent_name=agent_name, env=env)
    # env = ChallengeWrapper(env=env, agent_name='Blue')
    return env


def get_model(rl_str: str, env: OpenAIGymWrapper):
    lr = 5e-4
    if rl_str == "PPO":
        return PPO("MlpPolicy", env, gamma=0.95,
                   learning_rate=lr,
                   n_steps=128,
                   batch_size=2048,
                   n_epochs=4,
                   ent_coef=0.025,
                   vf_coef=0.005,
                   clip_range=0.2,
                   clip_range_vf=None,
                   gae_lambda=0.9,
                   verbose=1,
                   tensorboard_log="PPO_log",
                   device="cuda",
                   seed=123)
    if rl_str == "DQN":
        return DQN("MlpPolicy", env, gamma=0.95,
                   learning_rate=lr, verbose=1, tensorboard_log="DQN_log", device="cuda",

                   )

    # checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=CHECKPOINTS_PATH,
    #                                          name_prefix='rl_model')


def run_episode(env: OpenAIGymWrapper):
    # fm = PPO.load(CHECKPOINTS_MODELS / "rl_model_30000_steps.zip")
    fm = PPO.load("trained_models/PPO_model.zip")
    # env = func(SCENARIO_PATH)
    done = True
    total_reward = 0
    for i in range(60):
        if done:
            np.random.seed(0)
            obs = env.reset()
            print("################## NEW EPISODE #########################")
        print(f"---------------- {env.step_counter} ------------------")
        action = fm.predict(obs)[0]
        s = str(env.env.env.reverse_possible_actions.iloc[action])
        obs, reward, done, info = env.step(action)
        total_reward += reward
        for c in ["Blue", "Green", "Red"]:
            # print(c, env.env.env.env.environment_controller.action[c])
            print(c, env.env.env.env.env.env.environment_controller.action[c])
        # print(env.env.env.env.environment_controller.reward)
        print(f"total: {total_reward:3.1f}, rewards: {env.env.env.env.env.env.environment_controller.reward}")
        #print(f"reward: {reward}")


if __name__ == '__main__':
    func = get_cw
    func = get_cw
    #np.set_printoptions(1)
    env = func(SCENARIO_PATH)
    rl_str = "PPO"
    # env = get_env(SCENARIO_PATH)
    # env = get_cw(SCENARIO_PATH)
    model = get_model(rl_str, env)
    #model.learn(total_timesteps=1000000)
    #model.save(f"trained_models/{rl_str}_{model.policy.features_dim}_model.zip")
    run_episode(env)

    # run_episode(func)

    # rl_train()
    # filename ="{}_{}.txt".format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), get_model_name())
    # #test checkpoints
    # for model in range(SAVE_FREQ, TIMESTEPS+1, SAVE_FREQ):
    #     res = rl_test(n_of_maps_to_test=200, save_video=False, checkpoint_model = model)
    #     with open(os.path.join(RES_PATH, filename), 'a') as f:
    #         f.write(res)
    # #create some videos from the final model
    # res = rl_test(n_of_maps_to_test=50, save_video=True, avoid_collisions=True)
    # with open(os.path.join(RES_PATH, filename), 'a') as f:
    #     f.write(res)
