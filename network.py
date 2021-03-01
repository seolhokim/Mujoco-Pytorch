import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self,state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.pi_fc1   = nn.Linear(state_dim,hidden_dim)
        self.pi_fc2   = nn.Linear(hidden_dim,hidden_dim)

        self.fc_pi = nn.Linear(hidden_dim,action_dim)
        self.fc_sigma = nn.Linear(hidden_dim,action_dim)
        
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()   
                
    def forward(self,x):
        x = F.tanh(self.pi_fc1(x))
        x = F.tanh(self.pi_fc2(x))
        mu =  F.tanh(self.fc_pi(x))
        sigma = F.softplus(self.fc_sigma(x)) +1e-3
        return mu,sigma

class Critic(nn.Module):
    def __init__(self,state_dim,hidden_dim):
        super(Critic, self).__init__()
        self.pi_fc1   = nn.Linear(state_dim,hidden_dim)
        self.pi_fc2   = nn.Linear(hidden_dim,hidden_dim)
        
        self.fc_v  = nn.Linear(hidden_dim,1)
        
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()       
                
    def forward(self,x):
        x = F.tanh(self.va_fc1(x))
        x = F.tanh(self.va_fc2(x))
        v = self.fc_v(x)
        return v
   
