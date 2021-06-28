# Mujoco-Pytorch
* Reinforcement learning algorithm Application on Mujoco

<left><img src="https://github.com/seolhokim/Mujoco-Pytorch/blob/main/assets/ant.gif" width="250" height="200"></left>
<left><img src="https://github.com/seolhokim/Mujoco-Pytorch/blob/main/assets/hopper.gif" width="250" height="200"></left>
<left><img src="https://github.com/seolhokim/Mujoco-Pytorch/blob/main/assets/swimmer.gif" width="250" height="200"></left>

## Implemented Algorithm
* PPO
* SAC
* DDPG
## Performance

### PPO

<left><img src="https://github.com/seolhokim/Mujoco-Pytorch/blob/main/assets/ppo_in_mujoco.PNG"></left>

#### hopper

<left><img src="https://github.com/seolhokim/Mujoco-Pytorch/blob/main/assets/hopper.PNG" width="300" height="300"></left>

#### reacher

<left><img src="https://github.com/seolhokim/Mujoco-Pytorch/blob/main/assets/reacher.PNG" width="300" height="300"></left>

#### swimmer

<left><img src="https://github.com/seolhokim/Mujoco-Pytorch/blob/main/assets/swimmer.PNG" width="300" height="300"></left>

#### walker2d

<left><img src="https://github.com/seolhokim/Mujoco-Pytorch/blob/main/assets/walker2d.PNG" width="300" height="300"></left>

#### humanoid

<left><img src="https://github.com/seolhokim/Mujoco-Pytorch/blob/main/assets/humanoid.PNG" width="300" height="300"></left>

#### ant

<left><img src="https://github.com/seolhokim/Mujoco-Pytorch/blob/main/assets/ant.PNG" width="300" height="300"></left>

#### halfcheetah

<left><img src="https://github.com/seolhokim/Mujoco-Pytorch/blob/main/assets/halfcheetah.PNG" width="300" height="300"></left>

## Requirements

- python 3.6. >=
- gym 0.17. >=
- mujoco_py 2.0. >=
- pytorch 1.6. >=

## RUN

~~~
python main.py
~~~

  - if you want to change any options, check "python main.py --help"
  - you train and test agent using main.py

- '--env_name', type=str, default = 'Hopper-v2', help = "'Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','HumanoidStandup-v2',\
          'InvertedDoublePendulum-v2', 'InvertedPendulum-v2','Walker2d-v2','Swimmer-v2','Reacher-v2'(default : Hopper-v2)"
- '--train', type=bool, default=True, help="(default: True)"
- '--render', type=bool, default=False, help="(default: False)"
- '--epochs', type=int, default=1000, help='number of epochs, (default: 1000)'
- '--entropy_coef', type=float, default=1e-2, help='entropy coef (default : 0.01)'
- '--critic_coef', type=float, default=0.5, help='critic coef (default : 0.5)'
- '--learning_rate', type=float, default=3e-4, help='learning rate (default : 0.0003)'
- '--gamma', type=float, default=0.99, help='gamma (default : 0.99)'
- '--lmbda', type=float, default=0.95, help='lambda using GAE(default : 0.95)'
- '--eps_clip', type=float, default=0.2, help='actor and critic clip range (default : 0.2)'
- '--K_epoch', type=int, default=64, help='train epoch number(default : 10)'
- '--T_horizon', type=int, default=2048, help='one generation before training(default : 2048)'
- '--hidden_dim', type=int, default=64, help='actor and critic network hidden dimension(default : 64)'
- '--minibatch_size', type=int, default=64, help='minibatch size(default : 64)'
- '--tensorboard', type=bool, default=False, help='use_tensorboard, (default: False)'
- '--load', type=str, default = 'no', help = 'load network name in ./model_weights'
- '--save_interval', type=int, default = 100, help = 'save interval(default: 100)'
- '--print_interval', type=int, default = 20, help = 'print interval(default : 20)'
- '--use_cuda', type=bool, default = True, help = 'cuda usage(default : True)'
