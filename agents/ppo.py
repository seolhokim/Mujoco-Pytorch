from networks.network import Actor, Critic
from utils.utils import ReplayBuffer, make_mini_batch, convert_to_tensor

import torch
import torch.nn as nn
import torch.optim as optim

class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, layer_num = 3, hidden_dim = 64,\
                 activation_function = torch.tanh, last_activation = None, trainable_std = True,\
                 actor_lr = 3e-4,critic_lr = 3e-4,entropy_coef = 1e-2,critic_coef =0.5, \
                 gamma = 0.99, lambd =0.95,max_clip= 0.2,train_epoch = 10,traj_length = 2048, batch_size = 64, max_grad_norm = 0.5, device = 'cpu'):
        super(PPO,self).__init__()
        
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.gamma = gamma
        self.lambd = lambd
        self.max_clip = max_clip
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.traj_length = traj_length
        self.data = ReplayBuffer(action_prob_exist = True, max_size = traj_length, state_dim = state_dim, num_action = action_dim)
        self.actor = Actor(layer_num, state_dim, action_dim, hidden_dim, \
                           activation_function,last_activation,trainable_std)
        self.critic = Critic(layer_num, state_dim, 1, hidden_dim, activation_function,last_activation)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.device = device
    def get_action(self,x):
        mu,sigma = self.actor(x)
        return mu,sigma
    
    def v(self,x):
        return self.critic(x)
    
    def put_data(self,transition):
        self.data.put_data(transition)
        
    def get_gae(self, states, rewards, next_states, done_masks):
        values = self.v(states).detach()
        td_target = rewards + self.gamma * self.v(next_states) * done_masks
        delta = td_target - values
        delta = delta.detach().cpu().numpy()
        advantage_lst = []
        advantage = 0.0
        for idx in reversed(range(len(delta))):
            if done_masks[idx] == 0:
                advantage = 0.0
            advantage = self.gamma * self.lambd * advantage + delta[idx][0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantages = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
        return values, advantages
    
    def train_net(self,n_epi,writer):
        data = self.data.sample(shuffle = False)
        states, actions, rewards, next_states, done_masks, old_log_probs = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'], data['log_prob'])
        
        old_values, advantages = self.get_gae(states, rewards, next_states, done_masks)
        returns = advantages + old_values
        advantages = (advantages - advantages.mean())/(advantages.std()+1e-3)
        
        for i in range(self.train_epoch):
            for state,action,old_log_prob,advantage,return_,old_value \
            in make_mini_batch(self.batch_size, states, actions, \
                                           old_log_probs,advantages,returns,old_values): 
                curr_mu,curr_sigma = self.get_action(state)
                value = self.v(state).float()
                curr_dist = torch.distributions.Normal(curr_mu,curr_sigma)
                entropy = curr_dist.entropy() * self.entropy_coef
                curr_log_prob = curr_dist.log_prob(action).sum(1,keepdim = True)

                #policy clipping
                ratio = torch.exp(curr_log_prob - old_log_prob.detach())
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.max_clip, 1+self.max_clip) * advantage
                actor_loss = (-torch.min(surr1, surr2) - entropy).mean() 
                
                #value clipping (PPO2 technic)
                old_value_clipped = old_value + (value - old_value).clamp(-self.max_clip,self.max_clip)
                value_loss = (value - return_.detach().float()).pow(2)
                value_loss_clipped = (old_value_clipped - return_.detach().float()).pow(2)
                critic_loss = 0.5 * self.critic_coef * torch.max(value_loss,value_loss_clipped).mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                if writer != None:
                    writer.add_scalar("loss/actor_loss", actor_loss.item(), n_epi)
                    writer.add_scalar("loss/critic_loss", critic_loss.item(), n_epi)