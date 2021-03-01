import gym
from utils import RunningMeanStd
from agent import PPO

import torch
from torch.utils.tensorboard import SummaryWriter

#Hyperparameters
entropy_coef  = 1e-2
critic_coef   = 0.5
learning_rate = 0.0003
gamma         = 0.99
lmbda         = 0.95
eps_clip      = 0.2
K_epoch       = 10
T_horizon     = 2048
hidden_space  = 64
minibatch_size = 64

env = gym.make("Hopper-v2")
action_space = env.action_space.shape[0]
state_space = env.observation_space.shape[0]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    agent = PPO(state_space,action_space,hidden_space, learning_rate,entropy_coef,critic_coef,gamma,lmbda,eps_clip,\
               K_epoch, minibatch_size).cuda()
else:
    agent = PPO(state_space,action_space)
state_rms = RunningMeanStd(state_space)

writer = SummaryWriter()
print_interval = 20
global_step = 0
render = False
score_lst = []

for n_epi in range(10000):
    score = 0.0
    s = (env.reset())
    s = np.clip((s - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
    done = False
    for t in range(T_horizon):
        global_step += 1 
        if render:    
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
            writer.add_scalar("score", score, n_epi)
            if score > max_score:
                #torch.save(agent.state_dict(),'./model_weights/agent_'+str(int(score))+"points")
                max_score = score
            score = 0
        else:
            s = s_prime
            
    agent.train_net(n_epi)
    
    if n_epi%print_interval==0 and n_epi!=0:
        print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst)/len(score_lst)))
        score_lst = []
