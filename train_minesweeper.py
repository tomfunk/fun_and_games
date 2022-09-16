import envs 
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import torch as th

check_env(envs.MinesweeperEnvIntermediate(), warn=True)

n_envs = 64
many_env = make_vec_env(envs.MinesweeperEnvIntermediate, n_envs=n_envs)
tb_log_name = 'PPO_2'
tensorboard_log = "./ppo_minesweeper_tensorboard/"

# policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[256, 256, 256], vf=[256, 256, 256])])
policy_kwargs = dict()

# train
# model = PPO(
#     'MlpPolicy', many_env, policy_kwargs=policy_kwargs,
#     tensorboard_log=tensorboard_log
# )
# model.learn(total_timesteps=50000000, tb_log_name=tb_log_name, eval_freq=1000, n_eval_episodes=10,)#, eval_env=eval_env, eval_freq=n_envs*100)
# model.save('ms_int_ppo1')

# retrain
model = PPO.load('ms_int_ppo2', env=many_env, tensorboard_log=tensorboard_log)
model.learn(total_timesteps=50000000, reset_num_timesteps=False, tb_log_name=tb_log_name)
model.save('ms_int_ppo3')
