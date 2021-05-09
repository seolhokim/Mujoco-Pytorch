import numpy as np
import torch

def make_transition(state,action,reward,next_state,done,log_prob=None):
    transition = {}
    transition['state'] = state
    transition['action'] = action
    transition['reward'] = reward
    transition['next_state'] = next_state
    transition['log_prob'] = log_prob
    transition['done'] = done
    return transition

class ReplayBuffer():
    def __init__(self, action_prob_exist, max_size, state_dim, num_action):
        self.max_size = max_size
        self.data_idx = 0
        self.action_prob_exist = action_prob_exist
        self.data = {}
        
        self.data['state'] = np.zeros((self.max_size, state_dim))
        self.data['action'] = np.zeros((self.max_size, num_action))
        self.data['reward'] = np.zeros((self.max_size, 1))
        self.data['next_state'] = np.zeros((self.max_size, state_dim))
        self.data['done'] = np.zeros((self.max_size, 1))
        if self.action_prob_exist :
            self.data['log_prob'] = np.zeros((self.max_size, 1))
    def put_data(self, transition):
        idx = self.data_idx % self.max_size
        self.data['state'][idx] = transition['state']
        self.data['action'][idx] = transition['action']
        self.data['reward'][idx] = transition['reward']
        self.data['next_state'][idx] = transition['next_state']
        done = transition['done']
        self.data['done'][idx] = 0.0 if done else 1.0
        if self.action_prob_exist :
            self.data['log_prob'][idx] = transition['log_prob']
        
        self.data_idx += 1
    def sample(self, batch_size, shuffle):
        if shuffle :
            sample_num = min(self.max_size, self.data_idx)
            rand_idx = np.random.choice(sample_num, batch_size,replace=False)
            sampled_data = {}
            sampled_data['state'] = self.data['state'][rand_idx]
            sampled_data['action'] = self.data['action'][rand_idx]
            sampled_data['reward'] = self.data['reward'][rand_idx]
            sampled_data['next_state'] = self.data['next_state'][rand_idx]
            sampled_data['done'] = self.data['done'][rand_idx]
            if self.action_prob_exist :
                sampled_data['log_prob'] = self.data['log_prob'][rand_idx]
            return sampled_data
        else:
            return self.data #[:batch_size]
    def size(self):
        return self.data_idx
    def choose_mini_batch(self, mini_batch_size, states, actions, rewards, next_states, done_mask, old_log_prob, advantages, returns, old_value):
        full_batch_size = len(states)
        full_indices = np.arange(full_batch_size)
        np.random.shuffle(full_indices)
        for i in range(full_batch_size // mini_batch_size):
            indices = full_indices[mini_batch_size*i : mini_batch_size*(i+1)]
            yield states[indices], actions[indices], rewards[indices], next_states[indices], done_mask[indices],\
                  old_log_prob[indices], advantages[indices], returns[indices],old_value[indices]