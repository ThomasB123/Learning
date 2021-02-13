
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
Retrieved from https://ojs.aaai.org/index.php/AAAI/article/view/11796)
'''

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
from common import *
#from common.layers import NoisyLinear
#from common.replay_buffer import ReplayBuffer
from IPython.display import clear_output
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
#from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

# hyperparameters
learning_rate = 0.001 #0.0005 # 0.001 in Rainbow
gamma         = 0.99 #0.98 # 0.99 in Rainbow
buffer_limit  = 10000 #50000 # 10000 in Rainbow
batch_size    = 32 # 32 in Rainbow
video_every   = 25
print_every   = 5

# adapted from a combination of:
# 1. https://github.com/akolishchak/doom-net-pytorch/blob/master/src/noisy_linear.py
# 2. https://github.com/Scitator/Run-Skeleton-Run/blob/master/common/modules/NoisyLinear.py
# 3. https://github.com/PacktPublishing/Hands-On-Game-AI-with-Python/blob/91732f5a739b792551ed610c54cc0e6168696311/Chapter_10/Chapter_10/common/layers.py#L7

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, use_cuda, bias=True, factorised=True, std_init=None):
        super(NoisyLinear, self).__init__()

        self.use_cuda = use_cuda
        self.in_features = in_features
        self.out_features = out_features
        self.factorised = factorised
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_sigma = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        if not std_init:
            if self.factorised:
                self.std_init = 0.4
            else:
                self.std_init = 0.017
        else:
            self.std_init = std_init
        self.reset_parameters(bias)

    def sample(self):
        if self.training:
            self.weight_epsilon.normal_()
            self.weight = self.weight_epsilon.mul(self.weight_sigma).add_(self.weight_mu)
            if self.bias is not None:
                self.bias_epsilon.normal_()
                self.bias = self.bias_epsilon.mul(self.bias_sigma).add_(self.bias_mu)
        else:
            self.weight = self.weight_mu.detach()
            if self.bias is not None:
                self.bias = self.bias_mu.detach()
        self.sampled = True

    def reset_parameters(self, bias):
        if self.factorised:
            mu_range = 1. / math.sqrt(self.weight_mu.size(1))
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
            if bias:
                self.bias_mu.data.uniform_(-mu_range, mu_range)
                self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
        else:
            mu_range = math.sqrt(3. / self.weight_mu.size(1))
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init)
            if bias:
                self.bias_mu.data.uniform_(-mu_range, mu_range)
                self.bias_sigma.data.fill_(self.std_init)
    
    def scale_noise(self, size):
        x = torch.Tensor(size).normal_()
        x = x.sign().mul(x.abs().sqrt())
        return x

    def forward(self, input):
        if self.factorised:
            epsilon_in = self.scale_noise(self.in_features)
            epsilon_out = self.scale_noise(self.out_features)
            weight_epsilon = Variable(epsilon_out.ger(epsilon_in))
            bias_epsilon = Variable(self.scale_noise(self.out_features))
        else:
            weight_epsilon = Variable(torch.Tensor(self.out_features, self.in_features).normal_())
            bias_epsilon = Variable(torch.Tensor(self.out_features).normal_())
        return F.linear(input,
                        self.weight_mu + self.weight_sigma.mul(weight_epsilon),
                        self.bias_mu + self.bias_sigma.mul(bias_epsilon))


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
env_id = 'CartPole-v0'
env = gym.make(env_id)

# Rainbow: Combining Improvements in Deep Reinforcement Learning
# code adapted from https://github.com/higgsfield/RL-Adventure/blob/master/7.rainbow%20dqn.ipynb

class RainbowDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, num_atoms, Vmin, Vmax):
        super(RainbowDQN, self).__init__()

        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.Vmin = Vmin # maybe not needed for Gravitar, but this is for cartpole
        self.Vmax = Vmax # maybe not needed for Gravitar, but this is for cartpole

        self.linear1 = nn.Linear(num_inputs, 32)
        self.linear2 = nn.Linear(64, self.num_atoms)

        self.noisy_value1 = NoisyLinear(64, 64, use_cuda=USE_CUDA)
        self.noisy_value2 = NoisyLinear(64, self.num_atoms, use_cuda=USE_CUDA)

        self.noisy_advantage1 = NoisyLinear(64, 64, use_cuda=USE_CUDA)
        self.noisy_advantage2 = NoisyLinear(64, self.num_atoms * self.num_actions, use_cuda=USE_CUDA)
        
        #self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256) # was here before
        #self.fc2 = nn.Linear(256, 84)
        #self.fc3 = nn.Linear(84, env.action_space.n)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)
        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)
        value = value.view(batch_size, 1, self.num_atoms)
        advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)
        x = value + advantage - advantage.mean(1, keepdim=True)
        x = F.softmax(x.view(-1, self.num_atoms)).view(-1, self.num_actions, self.num_atoms)

        #x = x.view(x.size(0),-1)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        return x
    
    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()
    
    def act(self, state):
        state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
        dist = self.forward(state).data.cpu()
        dist = dist * torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
        action = dist.sum(2).max(1)[1].numpy()[0]
        return action
    '''
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()
    '''

num_atoms = 51
Vmin = -10
Vmax = 10

current_model = RainbowDQN(env.observation_space.shape[0], env.action_space.n, num_atoms, Vmin, Vmax)
target_model = RainbowDQN(env.observation_space.shape[0], env.action_space.n, num_atoms, Vmin, Vmax)

if USE_CUDA:
    current_model = current_model.cuda()
    target_model = target_model.cuda()

optimiser = optim.Adam(current_model.parameters(), learning_rate)

replay_buffer = ReplayBuffer()

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())
update_target(current_model, target_model)

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

# Computing Temporal Difference Loss:

def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
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

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    current_model.reset_noise()
    target_model.reset_noise()
    
    return loss

def plot(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()

# Training:

num_frames = 15000

losses = []
all_rewards = []
episode_reward = 0

state = env.reset()
for frame_idx in range(1, num_frames+1):
    action = current_model.act(state)
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    state = next_state
    episode_reward += reward
    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
    if len(replay_buffer) > batch_size:
        loss = compute_td_loss(batch_size)
        losses.append(loss.data[0])
    if frame_idx % 200 == 0:
        plot(frame_idx, all_rewards, losses)
    if frame_idx % 1000 == 0:
        update_target(current_model, target_model)


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

"""**Train**

← You can download the videos from the videos folder in the files on the left
"""

# setup the Gravitar ram environment, and record a video every 50 episodes. You can use the non-ram version here if you prefer
# Gravitar-ram-v0
# Gravitar-ram-v4
# Gravitar-ramDeterministic-v0
# Gravitar-ramDeterministic-v4
# Gravitar-ramNoFrameskip-v0
# Gravitar-ramNoFrameskip-v4
# Gravitar-v0
# Gravitar-v4
# Gravitar-Deterministic-v0
# Gravitar-Deterministic-v4
# GravitarNoFrameskip-v0
# GravitarNoFrameskip-v4

#env = gym.make('Gravitar-ram-v0')
#env = gym.make('PongNoFrameskip-v4')
#env = gym.wrappers.Monitor(env, "./video", video_callable=lambda episode_id: (episode_id%video_every)==0,force=True)
'''
# reproducible environment and action spaces, do not change lines 6-11 here (tools > settings > editor > show line numbers)
seed = 742
torch.manual_seed(seed)
env.seed(seed)
random.seed(seed)
np.random.seed(seed)
env.action_space.seed(seed)

q = RainbowDQN()
q_target = RainbowDQN()
q_target.load_state_dict(q.state_dict())
memory = ReplayBuffer()

score    = 0.0
marking  = []
optimizer = optim.Adam(q.parameters(), lr=learning_rate)

for n_episode in range(int(1e32)):
    epsilon = max(0.01, 0.08 - 0.01*(n_episode/200)) # linear annealing from 8% to 1%
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

    # do not change lines 44-48 here, they are for marking the submission log
    marking.append(score)
    if n_episode%100 == 0:
        print("marking, episode: {}, score: {:.1f}, mean_score: {:.2f}, std_score: {:.2f}".format(
            n_episode, score, np.array(marking).mean(), np.array(marking).std()))
        marking = []

    # you can change this part, and print any data you like (so long as it doesn't start with "marking")
    if n_episode%print_every==0 and n_episode!=0:
        q_target.load_state_dict(q.state_dict())
        print("episode: {}, score: {:.1f}, epsilon: {:.2f}".format(n_episode, score, epsilon))

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
'''