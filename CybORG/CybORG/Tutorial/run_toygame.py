import sys

sys.path.append('.')
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, EvalCallback

from CybORG.Agents.Wrappers import *
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



def get_env(path: str) -> MyChallengeWrapper:
    agents = {
        'Red': B_lineAgent,
        'Green': GreenAgent
    }

    env = CybORG(path, 'sim', agents=agents)
    eer = EnumActionWrapper(env)
    wrappers = FixedFlatWrapper(eer)
    #env = OpenAIGymWrapper(env=wrappers, agent_name='Blue')
    env = MyChallengeWrapper(env=wrappers,agent_name="Blue")

    # env = table_wrapper(env, output_mode='vector')
    # env = EnumActionWrapper(env)
    # env = OpenAIGymWrapper(agent_name=agent_name, env=env)
    #env = ChallengeWrapper(env=env, agent_name='Blue')
    return env


def rl_train(env):
    model = PPO("MlpPolicy", env, gamma=0.95,
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
                seed=123)
    # model_path_and_name = os.path.join(MODELS_PATH, get_model_name()+'.zip')
    # model_path_and_name = os.path.join(MODELS_PATH, "randommap_100000000steps_lr0.0005_stepsize3_withenemy")
    # model = PPO.load(model_path_and_name, env=env,  custom_objects={'seed': np.random.randint(2 ** 20)})
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=CHECKPOINTS_PATH,
                                             name_prefix='rl_model')
    # eval_callback = EvalCallback(eval_env=get_env(SCENARIO_PATH), best_model_save_path="./logs/",
    #                              log_path="./logs/", eval_freq=500,
    #                              deterministic=True, render=False)
    # checkpoint_callback = CustomCallback()
    # print("start training. model path and name: {}".format(model_path_and_name))
    model.learn(total_timesteps=100000, log_interval=20, callback=checkpoint_callback)
    model.save(Path("trained_models") / "first_model.zip")


def run_episode(fname):
    fm = PPO.load(CHECKPOINTS_MODELS / "rl_model_30000_steps.zip")
    env = get_env(SCENARIO_PATH)
    done = True
    reward = 0
    for i in range(100):
        if done:
            np.random.seed(0)
            obs = env.reset()
            print("################## NEW EPISODE #########################")
        print(f"---------------- {env.step_counter} ------------------")
        action = fm.predict(obs)[0]
        s = str(env.env.env.reverse_possible_actions.iloc[action])
        obs, reward, done, info = env.step(action)
        for c in ["Blue","Green","Red"]:
            print(c,env.env.env.env.environment_controller.action[c])
        print(env.env.env.env.environment_controller.reward)
        print(f"reward: {reward}")


if __name__ == '__main__':
    env = get_env(SCENARIO_PATH)
    rl_train(env)
    #run_episode(SCENARIO_PATH)

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
