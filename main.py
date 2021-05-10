import gym
import numpy as np
import argparse
import os

from agent import PPO
from environment import NormalizedGymEnv
from utils import make_transition

import torch
os.makedirs('./model_weights', exist_ok=True)

parser = argparse.ArgumentParser('parameters')
parser.add_argument("--env_name", type=str, default = 'Hopper-v2', help = "'Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','HumanoidStandup-v2',\
          'InvertedDoublePendulum-v2', 'InvertedPendulum-v2' (default : Hopper-v2)")
parser.add_argument('--train', type=bool, default=True, help="(default: True)")
parser.add_argument('--render', type=bool, default=False, help="(default: False)")
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs, (default: 1000)')

parser.add_argument('--entropy_coef', type=float, default=1e-2, help='entropy coef (default : 0.01)')
parser.add_argument('--critic_coef', type=float, default=0.5, help='critic coef (default : 0.5)')
parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate (default : 0.0003)')
parser.add_argument('--gamma', type=float, default=0.99, help='gamma (default : 0.99)')
parser.add_argument('--lmbda', type=float, default=0.95, help='lambda using GAE(default : 0.95)')
parser.add_argument('--eps_clip', type=float, default=0.2, help='actor and critic clip range (default : 0.2)')
parser.add_argument('--K_epoch', type=int, default=10, help='train epoch number(default : 10)')
parser.add_argument('--T_horizon', type=int, default=2048, help='one generation before training(default : 2048)')
parser.add_argument('--hidden_dim', type=int, default=64, help='actor and critic network hidden dimension(default : 64)')
parser.add_argument('--minibatch_size', type=int, default=64, help='minibatch size(default : 64)')
parser.add_argument('--max_grad_norm', type=float, default=0.5, help='network gradient clipping (default : 0.5)')
parser.add_argument('--tensorboard', type=bool, default=False, help='use_tensorboard, (default: False)')
parser.add_argument("--load", type=str, default = 'no', help = 'load network name in ./model_weights')
parser.add_argument("--save_interval", type=int, default = 100, help = 'save interval(default: 100)')
parser.add_argument("--print_interval", type=int, default = 20, help = 'print interval(default : 20)')
parser.add_argument("--use_cuda", type=bool, default = True, help = 'cuda usage(default : True)')
args = parser.parse_args()


env_lst = ['Ant-v2','HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'HumanoidStandup-v2',\
          'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Walker2d-v2', 'Swimmer-v2', 'Reacher-v2']

assert args.env_name in env_lst

env = NormalizedGymEnv(args.env_name,normalize_obs=True)
'''
#for pybullet envs
import pybullet_envs
env = NormalizedGymEnv("HopperBulletEnv-v0",normalize_obs=True)
'''
action_space = env.action_space.shape[0]
state_space = env.observation_space.shape[0]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.use_cuda == False:
    device = 'cpu'
agent = PPO(state_space,action_space,args.hidden_dim, args.learning_rate,args.entropy_coef,args.critic_coef,args.gamma,args.lmbda,args.eps_clip,\
               args.K_epoch, args.T_horizon, args.minibatch_size, args.max_grad_norm, device)

if (torch.cuda.is_available()) and (args.use_cuda):
    agent = agent.cuda()

if args.load != 'no':
    agent.load_state_dict(torch.load("./model_weights/"+args.load))

if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
else:
    writer = None

score_lst = []

score = 0.0
state = (env.reset())
for n_epi in range(args.epochs):
    for t in range(args.T_horizon):
        if args.render:    
            env.render()
        mu,sigma = agent.pi(torch.from_numpy(state).float().to(device))
        dist = torch.distributions.Normal(mu,sigma[0])
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1,keepdim = True)
        next_state, reward, done, info = env.step(action.cpu().numpy())
        transition = make_transition(state,\
                                     action,\
                                     np.array([reward/10.0]),\
                                     next_state,\
                                     np.array([done]),\
                                     log_prob.detach().cpu().numpy()\
                                    )
        agent.put_data(transition) 
        score += reward
        if done:
            state = (env.reset())
            score_lst.append(score)
            if args.tensorboard:
                writer.add_scalar("score", score, n_epi)
            score = 0
        else:
            state = next_state
            
    agent.train_net(n_epi,writer)
    if n_epi%args.print_interval==0 and n_epi!=0:
        print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst)/len(score_lst)))
        score_lst = []
    if n_epi%args.save_interval==0 and n_epi!=0:
        torch.save(agent.state_dict(),'./model_weights/agent_'+str(n_epi))