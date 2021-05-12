from configparser import ConfigParser
from argparse import ArgumentParser

import torch
import gym
import numpy as np
import os

from agents.ppo import PPO
from agents.sac import SAC

from utils.environment import NormalizedGymEnv
from utils.utils import make_transition, get_value
os.makedirs('./model_weights', exist_ok=True)

parser = ArgumentParser('parameters')

parser.add_argument("--env_name", type=str, default = 'Hopper-v2', help = "'Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','HumanoidStandup-v2',\
          'InvertedDoublePendulum-v2', 'InvertedPendulum-v2' (default : Hopper-v2)")
parser.add_argument("--algo", type=str, default = 'ppo', help = 'algorithm to adjust (default : ppo)')
parser.add_argument('--train', type=bool, default=True, help="(default: True)")
parser.add_argument('--render', type=bool, default=False, help="(default: False)")
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs, (default: 1000)')
parser.add_argument('--normalize_obs', type=bool, default=False, help='normalize state, (default: False)')
parser.add_argument('--tensorboard', type=bool, default=False, help='use_tensorboard, (default: False)')
parser.add_argument("--load", type=str, default = 'no', help = 'load network name in ./model_weights')
parser.add_argument("--save_interval", type=int, default = 100, help = 'save interval(default: 100)')
parser.add_argument("--print_interval", type=int, default = 1, help = 'print interval(default : 20)')
parser.add_argument("--use_cuda", type=bool, default = True, help = 'cuda usage(default : True)')
args = parser.parse_args()
parser = ConfigParser()
parser.read('config.ini')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.use_cuda == False:
    device = 'cpu'

env = NormalizedGymEnv(args.env_name, normalize_obs = args.normalize_obs)
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]

if args.algo == 'ppo' :
    agent = PPO(state_dim,action_dim,\
                get_value(parser,args.algo, 'hidden_dim'), get_value(parser,args.algo, 'actor_lr'),\
                get_value(parser,args.algo, 'critic_lr'),get_value(parser,args.algo,'entropy_coef'),\
                get_value(parser,args.algo,'critic_coef'),get_value(parser,args.algo,'gamma'),\
                get_value(parser,args.algo,'lambda'),get_value(parser,args.algo,'max_clip'),\
                get_value(parser,args.algo,'train_epoch'), get_value(parser,args.algo,'traj_length'),\
                get_value(parser,args.algo,'batch_size'), get_value(parser,args.algo,'max_grad_norm'),\
                device)
elif args.algo == 'sac' :
    agent = SAC(state_dim, action_dim, get_value(parser,args.algo, 'hidden_dim'))

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

if get_value(parser,args.algo, 'on_policy') == True:
    score = 0.0
    state = (env.reset())
    for n_epi in range(args.epochs):
        for t in range(get_value(parser,args.algo,'traj_length')):
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
            
else : # off policy
    for n_epi in range(args.epochs):
        score = 0.0
        state = env.reset()
        done = False
        while not done:
            if args.render:    
                env.render()
            action, _ = agent.get_action(torch.from_numpy(state).float().to(device))
            action = action.cpu().detach().numpy()
            next_state, reward, done, info = env.step(action)

            transition = make_transition(state,\
                                         action,\
                                         np.array([reward/10.0]),\
                                         next_state,\
                                         np.array([done])\
                                        )
            agent.put_data(transition) 

            state = next_state

            score += reward
            if agent.data.data_idx > get_value(parser,args.algo, 'learn_start_size'): 
                agent.train_net(get_value(parser,args.algo, 'batch_size'), writer, n_epi)  
        score_lst.append(score)
        if args.tensorboard:
            writer.add_scalar("score/score", score, n_epi)
        if n_epi%args.print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst)/len(score_lst)))
            score_lst = []
        if n_epi%args.save_interval==0 and n_epi!=0:
            torch.save(agent.state_dict(),'./model_weights/agent_'+str(n_epi))