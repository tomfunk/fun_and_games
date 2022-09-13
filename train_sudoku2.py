import envs 
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import torch as th

check_env(envs.SudokuEnv2.create_very_easy(), warn=True)

n_envs = 32
many_env = make_vec_env(envs.SudokuEnv2.create_very_easy, n_envs=n_envs)

# policy_kwargs = dict(net_arch=[dict(pi=[256, 256, 256], vf=[256, 256, 256])])
policy_kwargs = dict()

model = PPO(
    'MlpPolicy', many_env, policy_kwargs=policy_kwargs,
    gamma=0.999,
    tensorboard_log="./ppo_sudoku_tensorboard/"
)
model.learn(total_timesteps=10000000, tb_log_name='PPO_2', eval_freq=1000, n_eval_episodes=10)
model.save('su2_ppo')

# model = PPO.load('ppo0', env=many_env, tensorboard_log="./ppo_sudoku_tensorboard/")
# model.learn(total_timesteps=8000000, reset_num_timesteps=False, tb_log_name='PPO_0')
# model.save('ppo0')
