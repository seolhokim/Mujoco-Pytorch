from networks.network import Actor, Critic
from utils.environment import NormalizedGymEnv
from utils.utils import ReplayBuffer, convert_to_tensor, make_transition

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal


class SAC(nn.Module):
    def __init__(self, state_dim, action_dim, layer_num, hidden_dim, \
                 activation_function, last_activation, trainable_std, alpha_init, \
                 gamma, q_lr, actor_lr, alpha_lr, soft_update_rate, device):
        super(SAC,self).__init__()
        self.actor = Actor(layer_num, state_dim, action_dim, hidden_dim, \
                           activation_function,last_activation,trainable_std)

        self.q_1 = Critic(layer_num, state_dim+action_dim, 1, hidden_dim, activation_function,last_activation)
        self.q_2 = Critic(layer_num, state_dim+action_dim, 1, hidden_dim, activation_function,last_activation)
        
        self.target_q_1 = Critic(layer_num, state_dim+action_dim, 1, hidden_dim, activation_function,last_activation)
        self.target_q_2 = Critic(layer_num, state_dim+action_dim, 1, hidden_dim, activation_function,last_activation)
        
        self.soft_update(self.q_1, self.target_q_1, 1.)
        self.soft_update(self.q_2, self.target_q_2, 1.)
        
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.data = ReplayBuffer(action_prob_exist = False, max_size = int(1e+6), state_dim = state_dim, num_action = action_dim)
        self.target_entropy = -torch.tensor(action_dim)
        
        self.gamma = gamma
        self.q_lr = q_lr
        self.actor_lr = actor_lr
        self.alpha_lr = alpha_lr
        self.soft_update_rate = soft_update_rate 
        self.device = device
        
        self.q_1_optimizer = optim.Adam(self.q_1.parameters(), lr=self.q_lr)
        self.q_2_optimizer = optim.Adam(self.q_2.parameters(), lr=self.q_lr)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.alpha_optimizer = optim.Adam([self.alpha], lr=self.alpha_lr)
        
    def put_data(self,transition):
        self.data.put_data(transition)
        
    def soft_update(self, network, target_network, rate):
        for network_params, target_network_params in zip(network.parameters(), target_network.parameters()):
            target_network_params.data.copy_(target_network_params.data * (1.0 - rate) + network_params.data * rate)
    
    def get_action(self,state):
        mu,std = self.actor(state)
        dist = Normal(mu, std)
        u = dist.rsample()
        u_log_prob = dist.log_prob(u)
        a = torch.tanh(u)
        a_log_prob = u_log_prob - torch.log(1 - torch.square(a) +1e-3)
        return a, a_log_prob.sum(-1, keepdim=True)
    
    def q_update(self, Q, q_optimizer, states, actions, rewards, next_states, done_masks):
        ###target
        with torch.no_grad():
            next_actions, next_action_log_prob = self.get_action(next_states)
            q_1 = self.target_q_1(next_states, next_actions)
            q_2 = self.target_q_2(next_states, next_actions)
            q = torch.min(q_1,q_2)
            v = done_masks * (q - self.alpha * next_action_log_prob)
            targets = rewards + self.gamma * v
        
        q = Q(states, actions)
        loss = F.smooth_l1_loss(q, targets)
        q_optimizer.zero_grad()
        loss.backward()
        q_optimizer.step()
        return loss
    
    def actor_update(self, states):
        now_actions, now_action_log_prob = self.get_action(states)
        q_1 = self.q_1(states, now_actions)
        q_2 = self.q_2(states, now_actions)
        q = torch.min(q_1, q_2)
        
        loss = (self.alpha.detach() * now_action_log_prob - q).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        return loss,now_action_log_prob
    
    def alpha_update(self, now_action_log_prob):
        loss = (- self.alpha * (now_action_log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()    
        loss.backward()
        self.alpha_optimizer.step()
        return loss
    
    def train_net(self, batch_size, writer, n_epi):
        data = self.data.sample(shuffle = True, batch_size = batch_size)
        states, actions, rewards, next_states, done_masks = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'])

        ###q update
        q_1_loss = self.q_update(self.q_1, self.q_1_optimizer, states, actions, rewards, next_states, done_masks)
        q_2_loss = self.q_update(self.q_2, self.q_2_optimizer, states, actions, rewards, next_states, done_masks)

        ### actor update
        actor_loss,prob = self.actor_update(states)
        
        ###alpha update
        alpha_loss = self.alpha_update(prob)
        
        self.soft_update(self.q_1, self.target_q_1, self.soft_update_rate)
        self.soft_update(self.q_2, self.target_q_2, self.soft_update_rate)
        if writer != None:
            writer.add_scalar("loss/q_1", q_1_loss, n_epi)
            writer.add_scalar("loss/q_2", q_2_loss, n_epi)
            writer.add_scalar("loss/actor", actor_loss, n_epi)
            writer.add_scalar("loss/alpha", alpha_loss, n_epi)
            