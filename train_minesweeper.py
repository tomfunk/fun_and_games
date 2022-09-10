import envs 
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import torch as th

check_env(envs.MinesweeperEnvBeginner(), warn=True)

n_envs = 1
many_env = make_vec_env(envs.MinesweeperEnvBeginner, n_envs=n_envs)

policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[256, 256, 256], vf=[256, 256, 256])])
# policy_kwargs = dict()

model = PPO(
    'MlpPolicy', many_env, policy_kwargs=policy_kwargs, verbose=1, n_steps=2, batch_size=2,
    gamma=0.9999,
    tensorboard_log="./ppo_minesweeper_tensorboard/"
)
model.learn(total_timesteps=15000, tb_log_name='PPO_0')#, eval_env=eval_env, eval_freq=n_envs*100)
model.save('ms_ppo0')

# model = PPO.load('ppo0', env=many_env, tensorboard_log="./ppo_minesweeper_tensorboard/")
# model.learn(total_timesteps=8000000, reset_num_timesteps=False, tb_log_name='PPO_0')
# model.save('ppo0')
