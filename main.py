import gym
import numpy as np
import argparse

from utils import RunningMeanStd
from agent import PPO

import torch

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
parser.add_argument('--K_epoch', type=int, default=64, help='train epoch number(default : 10)')
parser.add_argument('--T_horizon', type=int, default=2048, help='one generation before training(default : 2048)')
parser.add_argument('--hidden_dim', type=int, default=64, help='actor and critic network hidden dimension(default : 64)')
parser.add_argument('--minibatch_size', type=int, default=64, help='minibatch size(default : 64)')
parser.add_argument('--tensorboard', type=bool, default=False, help='use_tensorboard, (default: False)')
parser.add_argument("--load", type=str, default = 'no', help = 'load network name in ./model_weights')
parser.add_argument("--save_interval", type=int, default = 100, help = 'save interval(default: 100)')
parser.add_argument("--print_interval", type=int, default = 20, help = 'print interval(default : 20)')
parser.add_argument("--use_cuda", type=bool, default = True, help = 'cuda usage(default : True)')
args = parser.parse_args()


env_lst = ['Ant-v2','HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'HumanoidStandup-v2',\
          'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Walker2d-v2', 'Swimmer-v2', 'Reacher-v2']

assert args.env_name in env_lst

env = gym.make(args.env_name)
action_space = env.action_space.shape[0]
state_space = env.observation_space.shape[0]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.use_cuda == False:
    device = 'cpu'
if (torch.cuda.is_available()) and (args.use_cuda):
    agent = PPO(state_space,action_space,args.hidden_dim, args.learning_rate,args.entropy_coef,args.critic_coef,args.gamma,args.lmbda,args.eps_clip,\
               args.K_epoch, args.minibatch_size,device).cuda()
else:
    agent = PPO(state_space,action_space,args.hidden_dim, args.learning_rate,args.entropy_coef,args.critic_coef,args.gamma,args.lmbda,args.eps_clip,\
               args.K_epoch, args.minibatch_size,device)

if args.load != 'no':
    agent.load_state_dict(torch.load("./model_weights/"+args.load))
state_rms = RunningMeanStd(state_space)

if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
else:
    writer = None

score_lst = []

for n_epi in range(args.epochs):
    score = 0.0
    s = (env.reset())
    s = np.clip((s - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
    for t in range(args.T_horizon):
        global_step += 1 
        if args.render:    
            env.render()
        mu,sigma = agent.pi(torch.from_numpy(s).float().to(device))
        dist = torch.distributions.Normal(mu,sigma)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1,keepdim = True)
        s_prime, r, done, info = env.step(action.unsqueeze(0).cpu().numpy())
        s_prime = np.clip((s_prime - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
        agent.put_data((s, action, r/10.0, s_prime, \
                        log_prob.detach().cpu().numpy(), done))
        score += r
        if done:
            s = (env.reset())
            s = np.clip((s - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
            score_lst.append(score)
            if args.tensorboard:
                writer.add_scalar("score", score, n_epi)
            score = 0
        else:
            s = s_prime
            
    agent.train_net(n_epi,state_rms,writer)
    
    if n_epi%args.print_interval==0 and n_epi!=0:
        print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst)/len(score_lst)))
        score_lst = []
    if n_epi%args.save_interval==0 and n_epi!=0:
        torch.save(agent.state_dict(),'./model_weights/agent_'+str(n_epi))
