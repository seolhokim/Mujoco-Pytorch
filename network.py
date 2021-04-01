import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class Actor(nn.Module):
    def __init__(self,state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1   = nn.Linear(state_dim,hidden_dim)
        self.fc2   = nn.Linear(hidden_dim,hidden_dim)

        self.pi = nn.Linear(hidden_dim,action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()   
                
    def forward(self,x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = self.pi(x)
        std = torch.exp(self.actor_logstd)
        return mu,std

class Critic(nn.Module):
    def __init__(self,state_dim,hidden_dim):
        super(Critic, self).__init__()
        self.fc1   = nn.Linear(state_dim,hidden_dim)
        self.fc2   = nn.Linear(hidden_dim,hidden_dim)
        
        self.v  = nn.Linear(hidden_dim,1)
        
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()       
                
    def forward(self,x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        v = self.v(x)
        return v
   
