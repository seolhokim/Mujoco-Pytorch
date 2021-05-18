from networks.network import Actor, Critic
from utils.utils import ReplayBuffer, make_mini_batch, convert_to_tensor

import torch
import torch.nn as nn
import torch.optim as optim

class PPO(nn.Module):
    def __init__(self, writer, device, state_dim, action_dim, args):
        super(PPO,self).__init__()
        self.args = args
        
        self.data = ReplayBuffer(action_prob_exist = True, max_size = self.args.traj_length, state_dim = state_dim, num_action = action_dim)
        self.actor = Actor(self.args.layer_num, state_dim, action_dim, self.args.hidden_dim, \
                           self.args.activation_function,self.args.last_activation,self.args.trainable_std)
        self.critic = Critic(self.args.layer_num, state_dim, 1, \
                             self.args.hidden_dim, self.args.activation_function,self.args.last_activation)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.args.critic_lr)

        self.writer = writer
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
        td_target = rewards + self.args.gamma * self.v(next_states) * done_masks
        delta = td_target - values
        delta = delta.detach().cpu().numpy()
        advantage_lst = []
        advantage = 0.0
        for idx in reversed(range(len(delta))):
            if done_masks[idx] == 0:
                advantage = 0.0
            advantage = self.args.gamma * self.args.lambda_ * advantage + delta[idx][0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantages = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
        return values, advantages
    
    def train_net(self,n_epi):
        data = self.data.sample(shuffle = False)
        states, actions, rewards, next_states, done_masks, old_log_probs = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'], data['log_prob'])
        
        old_values, advantages = self.get_gae(states, rewards, next_states, done_masks)
        returns = advantages + old_values
        advantages = (advantages - advantages.mean())/(advantages.std()+1e-3)
        
        for i in range(self.args.train_epoch):
            for state,action,old_log_prob,advantage,return_,old_value \
            in make_mini_batch(self.args.batch_size, states, actions, \
                                           old_log_probs,advantages,returns,old_values): 
                curr_mu,curr_sigma = self.get_action(state)
                value = self.v(state).float()
                curr_dist = torch.distributions.Normal(curr_mu,curr_sigma)
                entropy = curr_dist.entropy() * self.args.entropy_coef
                curr_log_prob = curr_dist.log_prob(action).sum(1,keepdim = True)

                #policy clipping
                ratio = torch.exp(curr_log_prob - old_log_prob.detach())
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.args.max_clip, 1+self.args.max_clip) * advantage
                actor_loss = (-torch.min(surr1, surr2) - entropy).mean() 
                
                #value clipping (PPO2 technic)
                old_value_clipped = old_value + (value - old_value).clamp(-self.args.max_clip,self.args.max_clip)
                value_loss = (value - return_.detach().float()).pow(2)
                value_loss_clipped = (old_value_clipped - return_.detach().float()).pow(2)
                critic_loss = 0.5 * self.args.critic_coef * torch.max(value_loss,value_loss_clipped).mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
                self.critic_optimizer.step()
                
                if self.writer != None:
                    self.writer.add_scalar("loss/actor_loss", actor_loss.item(), n_epi)
                    self.writer.add_scalar("loss/critic_loss", critic_loss.item(), n_epi)