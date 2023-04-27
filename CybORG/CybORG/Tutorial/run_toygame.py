import yaml
import gym
import logging
import sys

sys.path.append('.')
from pathlib import Path

import numpy as np
from enum import Enum
# import imageio
import os
import matplotlib.pyplot as plt
import time
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, EvalCallback
import datetime

from CybORG.Agents.Wrappers.ChallengeWrapper import ChallengeWrapper
from CybORG import CybORG

from CybORG.Agents import B_lineAgent, GreenAgent, BlueMonitorAgent

# from hstoy_env.envs.envs_dir.hstoy_env import *

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


# class EnvNum():
#     def __init__(self):
#         self.n = -1
#
#     def get(self):
#         self.n += 1
#         return self.n
#
# def get_model_name():
#     return 'randommap_{}steps_lr{}_stepsize{}_3enemies_rot'.format(TIMESTEPS, lr, STEP_SIZE)

def get_env(path: str) -> ChallengeWrapper:
    agents = {
        'Red': B_lineAgent,
        'Green': GreenAgent
    }

    env = CybORG(path, 'sim', agents=agents)
    env = ChallengeWrapper(env=env, agent_name='Blue')
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


# def rl_test(n_of_maps_to_test, save_video = False, checkpoint_model = None, avoid_collisions = False):
#     if checkpoint_model is None:
#         model_path_and_name = os.path.join(MODELS_PATH, get_model_name()+'.zip')
#     else:
#         model_path_and_name = os.path.join(CHECKPOINTS_PATH, "rl_model_{}_steps".format(checkpoint_model))
#     print("\ntesting model {}".format(model_path_and_name))
#     model = PPO.load(model_path_and_name, custom_objects={'seed': np.random.randint(2 ** 20)})
#     env = HSToyEnv()  # gym.make(HSToyEnv)
#     all_steps =0
#     acc_coverage = 0
#     acc_distances = 0
#     acc_on_building = 0
#     n_success =0
#     acc_enemy_killed=0
#     acc_killed_by_enemy=0
#     for i in range(n_of_maps_to_test):
#
#         obs = env.reset()
#         info_exists = False
#         images = []
#         n_steps = 0
#         if save_video:
#             img = env.render(mode='rgb',reward= 0)
#             for _ in range(25):
#                 images.append(img)
#         while True:
#             # unfortunately reset does not expose info. Hence immediately after reset we cannot check the action
#             if avoid_collisions and info_exists:
#                 # if action hits a building or takes out of boundaries choose the next best action
#                 probs = []
#                 ob = model.policy.obs_to_tensor(obs)[0]
#                 dist = model.policy.get_distribution(ob)
#                 for d in dist.distribution:
#                     probs.append(d.probs.detach().cpu().numpy()[0])
#                 P = 1
#                 for p in probs:
#                     P = np.multiply.outer(P,p)
#                 ordered_actions = np.array(np.unravel_index(np.argsort(P, axis=None)[::-1], P.shape)).T
#                 for action in ordered_actions:
#                     if not forbidden_action(action, info):
#                         break
#             else:
#                 action, _ = model.predict(obs, deterministic=True)
#             obs, reward, dones, info = env.step(action)
#             info_exists = True
#             if save_video:
#                 img = env.render(mode='rgb', reward=reward)
#                 images.append(img)
#
#             n_steps += 1
#             if dones:
#
#                 all_steps += n_steps
#                 acc_coverage += info["coverage"]
#                 acc_distances += info["distance"]
#                 acc_on_building += info["on_building"]
#                 acc_enemy_killed += info["enemy_killed"]
#                 acc_killed_by_enemy += info["killed_by_enemy"]
#                 if info["finished"]:
#                     n_success += 1
#                 if save_video:
#                     for _ in range(25):
#                         images.append(img)
#                     filename = os.path.join(VIDEO_PATH, get_model_name() + '_test%d.mp4' % (i))
#                     print("video path:", filename)
#                     imageio.mimsave(filename, images)
#                 break
#     str_res = "model {}. success in {:3d} out of {:3d} tests. success rate {:5.2f}%. Averaged coverage {:5.2f}%. " \
#               "Averaged number of steps (including failures) {:6.2f}. Average distance {:6.1f}. Steps on building {:5.2f} " \
#               "Killed {} enemies. Killed by {} enemies.\n"\
#         .format(checkpoint_model if checkpoint_model else get_model_name(), n_success, n_of_maps_to_test,
#                 100*n_success / n_of_maps_to_test, acc_coverage / n_of_maps_to_test, all_steps / n_of_maps_to_test,
#                 acc_distances / n_of_maps_to_test, acc_on_building / n_of_maps_to_test,
#                 acc_enemy_killed / n_of_maps_to_test, acc_killed_by_enemy / n_of_maps_to_test)
#     print(str_res)
#     return str_res
#
# def forbidden_action(a, info):
#     head_angle = (info["head"] + a[1] - HEAD_MAX_STEP_ANGLE) % N_POSSIBLE_HEAD_ANGLES
#     if head_angle == 0: dy, dx = -a[0], 0
#     if head_angle == 1: dy, dx = -a[0], a[0]
#     if head_angle == 2: dy, dx = 0, a[0]
#     if head_angle == 3: dy, dx = a[0], a[0]
#     if head_angle == 4: dy, dx = a[0], 0
#     if head_angle == 5: dy, dx = a[0], -a[0]
#     if head_angle == 6: dy, dx = 0, -a[0]
#     if head_angle == 7: dy, dx = -a[0], -a[0]
#     y,x = info["position"]
#     y += dy
#     x += dx
#     if y < 0 or x < 0 or y >= HEIGHT or x >= WIDTH:
#         return True
#     if (info["map"][y,x] == COLORS_DICT["BUILDING"]) or (info["map"][y,x] == COLORS_DICT["WINDOW"]):
#         return True
#     return False


if __name__ == '__main__':
    env = get_env(SCENARIO_PATH)
    rl_train(env)

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
