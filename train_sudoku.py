import envs 
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import torch as th

check_env(envs.SudokuEnv1.create_very_easy(), warn=True)

n_envs = 8
many_env = make_vec_env(envs.SudokuEnv1.create_very_easy, n_envs=n_envs)

# policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[256, 256, 256], vf=[256, 256, 256])])
policy_kwargs = dict()

model = PPO(
    'MlpPolicy', many_env, policy_kwargs=policy_kwargs, verbose=1, n_steps=2, batch_size=n_envs*2,
    gamma=0.95, learning_rate=0.05,
    tensorboard_log="./ppo_sudoku_tensorboard/"
)
model.learn(total_timesteps=1000000, tb_log_name='PPO_0')#, eval_env=eval_env, eval_freq=n_envs*100)
model.save('su_ppo0')

# model = PPO.load('ppo0', env=many_env, tensorboard_log="./ppo_sudoku_tensorboard/")
# model.learn(total_timesteps=8000000, reset_num_timesteps=False, tb_log_name='PPO_0')
# model.save('ppo0')
