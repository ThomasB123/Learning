
# gravitar

# this is a Deep Q Learning (DQN) agent including replay memory and a target network 
# you can write a brief 8-10 line abstract detailing your submission and experiments here
# the code is based on https://github.com/seungeunrho/minimalRL/blob/master/dqn.py, which is released under the MIT licesne
# make sure you reference any code you have studied as above, with one comment line per reference
'''
Abstract:
Rainbow combines 7 x DQN Extensions ( A3C, DQN, DDQN, Prioritised DDQN,
Dueling DDQN, Distributional DQN, and Noisy DQN )
We have focused here on value-based methods in the Q-learning family.
We have not considered purely policy based RL algorithms
From experimentation in the cited paper, ablation of Noisy DQN is indicated as increasing performance on this specific game, the largest gains appear to be from A3C and Prioritised DDQN.
( Hessel, M., Modayil, J., van Hasselt, H., Schaul, T., Ostrovski, G., Dabney, W., Horgan, D., Piot, B., Azar, M., 
& Silver, D. (2018). Rainbow: Combining Improvements in Deep Reinforcement Learning. 
Proceedings of the AAAI Conference on Artificial Intelligence, 32(1). 
Retrieved from https://arxiv.org/abs/1710.02298
'''
# the code is based on https://github.com/higgsfield/RL-Adventure
# best Gravitar results come from no noisy, no dueling, no double, 
# but using priority, multi-step, distribution, with biggest impact from 
# multi-step (https://arxiv.org/abs/1710.02298)

# imports
import gym
import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import torch.autograd as autograd
from IPython.display import clear_output
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from collections import deque
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)
import operator

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

# Rainbow: Combining Improvements in Deep Reinforcement Learning
# code adapted from https://github.com/higgsfield/RL-Adventure/blob/master/7.rainbow%20dqn.ipynb

class RainbowCnnDQN(nn.Module):
    def __init__(self):#, input_shape, num_actions, num_atoms, Vmin, Vmax):
        super(RainbowCnnDQN, self).__init__()

        #self.input_shape = input_shape
        #self.num_actions = num_actions
        #self.num_atoms = num_atoms
        #self.Vmin = Vmin
        #self.Vmax = Vmax

        #self.features = nn.Sequential(
        #    nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
        #    nn.ReLU(),
        #    nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #    nn.ReLU(),
        #    nn.Conv2d(64, 64, kernel_size=3, stride=1),
        #    nn.ReLU()
        #)

        #self.noisy_value1 = NoisyLinear(self.feature_size(), 512, use_cuda=USE_CUDA)
        #self.noisy_value2 = NoisyLinear(512, self.num_atoms, use_cuda=USE_CUDA)

        #self.noisy_advantage1 = NoisyLinear(self.feature_size(), 512, use_cuda=USE_CUDA)
        #self.noisy_advantage2 = NoisyLinear(512, self.num_atoms * self.num_actions, use_cuda=USE_CUDA)
        
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256) # was here before
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, env.action_space.n)

    def forward(self, x):
        #batch_size = x.size(0)
        #x = x / 255.
        #x = self.features(x)
        #x = x.view(batch_size, -1)

        #value = F.relu(self.noisy_value1(x))
        #value = self.noisy_value2(value)

        #advantage = F.relu(self.noisy_advantage1(x))
        #advantage = self.noisy_advantage2(advantage)

        #value = value.view(batch_size, 1, self.num_atoms)
        #advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)

        #x = value + advantage - advantage.mean(1, keepdim=True)
        #x = F.softmax(x.view(-1, self.num_atoms)).view(-1, self.num_actions, self.num_atoms)

        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    #def reset_noise(self):
    #    self.noisy_value1.reset_noise()
    #    self.noisy_value2.reset_noise()
    #    self.noisy_advantage1.reset_noise()
    #    self.noisy_advantage2.reset_noise()
    
    #def feature_size(self):
    #    return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    #def act(self, state):
    #    state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
    #    dist = self.forward(state).data.cpu()
    #    dist = dist * torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
    #    action = dist.sum(2).max(1)[1].numpy()[0]
    #    return action
    
    def sample_action(self, obs, epsilon): # sample action or take random action according with probability epsilon
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()
'''
def projection_distribution(next_state, rewards, dones):
    batch_size = next_state.size(0)
    
    delta_z = float(Vmax - Vmin) / (num_atoms - 1)
    support = torch.linspace(Vmin, Vmax, num_atoms)
    
    next_dist = target_model(next_state).data.cpu() * support
    next_action = next_dist.sum(2).max(1)[1]
    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
    next_dist = next_dist.gather(1, next_action).squeeze(1)
    
    rewards = rewards.unsqueeze(1).expand_as(next_dist)
    dones = dones.unsqueeze(1).expand_as(next_dist)
    support = support.unsqueeze(0).expand_as(next_dist)

    Tz = rewards + (1 - dones) * 0.99 * support
    Tz = Tz.clamp(min=Vmin, max=Vmax)
    b = (Tz - Vmin / delta_z)
    l = b.floor().long()
    u = b.ceil().long()

    offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long().unsqueeze(1).expand(batch_size, num_atoms)
    proj_dist = torch.zeros(next_dist.size())
    proj_dist.view(-1).index_add(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
    proj_dist.view(-1).index_add(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
    return proj_dist
'''
# Computing Temporal Difference Loss:
'''
def compute_td_loss(batch_size):
    state, action, reward, next_state, done, _, _ = replay_buffer.sample(batch_size,1)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))#, volatile=True)
    action = Variable(torch.LongTensor(action))
    reward = torch.FloatTensor(reward)
    done = torch.FloatTensor(np.float32(done))

    proj_dist = projection_distribution(next_state, reward, done)

    dist = current_model(state)
    action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_atoms)
    dist = dist.gather(1, action).squeeze(1)
    dist.data.clamp(0.01, 0.99)
    loss = -(Variable(proj_dist) * dist.log()).sum(1)
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    current_model.reset_noise()
    target_model.reset_noise()
    
    return loss
'''
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# hyperparameters
learning_rate = 0.0000625 # Adam learning rate # from https://arxiv.org/pdf/1710.02298.pdf, DQN uses 0.00025
gamma         = 0.99 # discount factor
buffer_limit  = 100000 # 1,000,000 used in https://arxiv.org/pdf/1710.02298.pdf
batch_size    = 32 # minibatch size
video_every   = 25
print_every   = 10
epsilon = 0.0 # exploration

num_atoms = 51 # distributional atoms
Vmin = -10 # distributional min
Vmax = 10 # distributional max

losses = []
all_rewards = []
episode_reward = 0

# setup the Gravitar ram environment, and record a video every 50 episodes. You can use the non-ram version here if you prefer
# Gravitar-ram-v0
# Gravitar-v0
env_id = 'Gravitar-v0'
env = gym.make('Gravitar-ram-v0')
env = gym.wrappers.Monitor(env, "./video", video_callable=lambda episode_id: (episode_id%video_every)==0,force=True)

# reproducible environment and action spaces, do not change lines 6-11 here (tools > settings > editor > show line numbers)
seed = 742
torch.manual_seed(seed)
env.seed(seed)
random.seed(seed)
np.random.seed(seed)
env.action_space.seed(seed)

q = RainbowCnnDQN()#env.observation_space.shape, env.action_space.n, num_atoms, Vmin, Vmax)
q_target = RainbowCnnDQN()#env.observation_space.shape, env.action_space.n, num_atoms, Vmin, Vmax)
q_target.load_state_dict(q.state_dict())
memory = ReplayBuffer()

if USE_CUDA:
    q = q.cuda()
    q_target = q_target.cuda()

#replay_initial = 10000
#replay_buffer = PrioritizedReplayBuffer(buffer_limit,1)

# q: model
# q_target: target_model
# s: state
# a: action
# r: reward
# s_prime: next_state

score    = 0.0
marking  = []
optimizer = optim.Adam(q.parameters(), lr=learning_rate)
#s = env.reset()
for n_episode in range(int(1e32)):
    #epsilon = max(0.01, 0.08 - 0.01*(n_episode/200)) # linear annealing from 8% to 1%
    s = env.reset()
    done = False
    score = 0.0

    while True:

        a = q.sample_action(torch.from_numpy(s).float().unsqueeze(0), epsilon)
        s_prime, r, done, info = env.step(a)
        done_mask = 0.0 if done else 1.0
        memory.put((s,a,r/100.0,s_prime, done_mask))
        s = s_prime

        score += r
        if done:
            break

    if memory.size()>2000:
        train(q, q_target, memory, optimizer)
    
    if n_episode % 1000 == 0:
        q_target.load_state_dict(q.state_dict())

    # do not change lines 44-48 here, they are for marking the submission log
    marking.append(score)
    if n_episode%100 == 0:
        print("marking, episode: {}, score: {:.1f}, mean_score: {:.2f}, std_score: {:.2f}".format(
            n_episode, score, np.array(marking).mean(), np.array(marking).std()))
        marking = []

    # you can change this part, and print any data you like (so long as it doesn't start with "marking")
    if n_episode%print_every==0 and n_episode!=0:
        q_target.load_state_dict(q.state_dict())
        print("episode: {}, score: {:.1f}".format(n_episode, score, epsilon))

# for coursework, assemble list of different techniques I can use and how they would fit together
# e.g. Dyna-Q, based on regular Q-learning but with additional inner loop simulation of environment
# rather than picking random s and a in for loop from slide 10, use Monto Carlo tree search to pick important observations, localised and prioratised
# look at taxonomy graph showing ho everything relates together
# also take any pseudocode from lectures and convert into python to help understand it and how it works
# best approach is probably combination of several from lectures
# can train on either RAM or pixels, RAM is feasible in ATARI since it's small and not encrypted,
# but also screen is not HD so pixels could be trained on as well

# not expecting human level performance
# possible to get scores of over 10 or 20,000
# but can get 1st scores with very low hundreds if you had a really good go at it and you've taken the content on board
# will not just apply a linear function for score
# to code dreamer in pytorch just use nn.grucell is a cell in this recurrent neural network, don't need to code it yourself
# learn dynamics using autoencoder and experience

### look at dreamer algorithm from lecture slides ###

# advice from Chris:
# approach overly complex papers more like an engineer
# run the code and dismantle it back down to the concepts that make it work
# lower variance can really help with lowering training times for iteration
