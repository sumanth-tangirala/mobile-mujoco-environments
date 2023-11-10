import numpy as np 
from factory import MushrEnvironmentFactory


env_factory = MushrEnvironmentFactory(
    max_speed=0.5,
    max_steering_angle=0.5,
    max_steps=100,
    prop_steps=10,
    goal_limits=[0, 5],
    with_obstacles=True,
)

env_name = "X2Obs"

checkpt_path = "./trained_models/"+env_name+"_0_5_100_10_obstacles_sac/"
load_path = "./trained_models/"+env_name+"_0_5_100_10_sac/"

load_model = True
train_model = True

# env_factory.register_environments_with_position_and_orientation_goals()
env_factory.register_environments_with_position_goals()

from stable_baselines3 import HER, SAC
import gym
import os
np.set_printoptions(suppress=True)

env = gym.make(env_name+"Env-v0")

breakpoint()
alg = SAC
num_steps = int(3e6)

model = HER('MlpPolicy', env, alg, n_sampled_goal=4,
            goal_selection_strategy='future',
            verbose=1, buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95, batch_size=256,max_episode_length=env.max_steps)

from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.her import ObsDictWrapper


eval_env = DummyVecEnv([lambda: gym.make(env_name+'Env-v0')])
eval_env = ObsDictWrapper(eval_env)

if not os.path.exists(checkpt_path):
    os.makedirs(checkpt_path)

if train_model:
    if load_model:
        model = HER.load(load_path + "/best/best_model", env=env, verbose=1)

    checkpoint_callback = CheckpointCallback(save_freq=10000,save_path=checkpt_path)
    eval_callback = EvalCallback(eval_env,n_eval_episodes=100,eval_freq=1e4,best_model_save_path=checkpt_path+"best/",
                            log_path=checkpt_path+"logs/",deterministic=True)
    callback = CallbackList([checkpoint_callback,eval_callback])
    model.learn(int(num_steps),callback=callback)
    model.save(checkpt_path)
else:
    model = SAC.load(checkpt_path+"/best/best_model", env=env)
