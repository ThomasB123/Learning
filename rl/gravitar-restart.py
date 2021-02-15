
# this is a Deep Q Learning (DQN) agent including replay memory and a target network 
# you can write a brief 8-10 line abstract detailing your submission and experiments here
# the code is based on https://github.com/seungeunrho/minimalRL/blob/master/dqn.py, which is released under the MIT licesne
# make sure you reference any code you have studied as above, with one comment line per reference

# imports
import gym
import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.autograd as autograd

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

class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions, num_atoms, Vmin, Vmax):
        super(QNetwork, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax

        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, env.action_space.n)

        #self.features = nn.Sequential(
        #    nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
        #    nn.ReLU(),
        #    nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #    nn.ReLU(),
        #    nn.Conv2d(64, self.num_actions, kernel_size=3, stride=1),
        #    nn.ReLU()
        #)

    def forward(self, x):
        #batch_size = x.size(0)
        #x = x / 255.
        #x = self.features(x)
        #x = x.view(batch_size,-1)
        #x = F.softmax(x.view(-1, self.num_atoms)).view(-1, self.num_actions, self.num_atoms)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs):#, epsilon):
        obs = Variable(torch.FloatTensor(np.float32(obs)).unsqueeze(0), volatile=True)
        dist = self.forward(obs).data.cpu()
        dist = dist * torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
        a = dist.sum(2).max(1)[1].numpy()[0]
        return a

def projection_distribution(s_prime, rs, dones):
    batch_size  = s_prime.size(0)
    
    delta_z = float(Vmax - Vmin) / (num_atoms - 1)
    support = torch.linspace(Vmin, Vmax, num_atoms)
    
    next_dist   = q_target(s_prime).data.cpu() * support
    next_action = next_dist.sum(2).max(1)[1]
    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
    next_dist   = next_dist.gather(1, next_action).squeeze(1)
        
    rs = rs.unsqueeze(1).expand_as(next_dist)
    dones   = dones.unsqueeze(1).expand_as(next_dist)
    support = support.unsqueeze(0).expand_as(next_dist)
    
    Tz = rs + (1 - dones) * 0.99 * support
    Tz = Tz.clamp(min=Vmin, max=Vmax)
    b  = (Tz - Vmin) / delta_z
    l  = b.floor().long()
    u  = b.ceil().long()
        
    offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long().unsqueeze(1).expand(batch_size, num_atoms)

    proj_dist = torch.zeros(next_dist.size())    
    proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
    proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
        
    return proj_dist

def train(q, q_target, memory, optimizer): # compute_td_loss
    s,a,r,s_prime,done_mask = memory.sample(batch_size)

    #q_out = q(s)
    #q_a = q_out.gather(1,a)
    #max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
    #target = r + gamma * max_q_prime * done_mask
    #loss = F.smooth_l1_loss(q_a, target)


    s = Variable(torch.FloatTensor(np.float32(s)))
    s_prime = Variable(torch.FloatTensor(np.float32(s_prime)), volatile=True)
    a = Variable(torch.LongTensor(a))
    r = torch.FloatTensor(r)
    done_mask = torch.FloatTensor(np.float32(done_mask))

    proj_dist = projection_distribution(s_prime, r, done)

    dist = q(s)
    action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_atoms)
    dist = dist.gather(1, a).squeeze(1)
    dist.data.clamp_(0.01, 0.99)
    loss = -(Variable(proj_dist) * dist.log()).sum(1)
    loss  = loss.mean()
            
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #current_model.reset_noise()
    #target_model.reset_noise()
        
    return loss

# hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32
video_every   = 25
print_every   = 5

num_atoms = 51
Vmin = -10
Vmax = 10


# setup the Gravitar ram environment, and record a video every 50 episodes. You can use the non-ram version here if you prefer
env = gym.make('Gravitar-ram-v0')
env = gym.wrappers.Monitor(env, "./video", video_callable=lambda episode_id: (episode_id%video_every)==0,force=True)

# reproducible environment and action spaces, do not change lines 6-11 here (tools > settings > editor > show line numbers)
seed = 742
torch.manual_seed(seed)
env.seed(seed)
random.seed(seed)
np.random.seed(seed)
env.action_space.seed(seed)

q = QNetwork(env.observation_space.shape, env.action_space.n, num_atoms, Vmin, Vmax)
q_target = QNetwork(env.observation_space.shape, env.action_space.n, num_atoms, Vmin, Vmax)
q_target.load_state_dict(q.state_dict())
memory = ReplayBuffer()

score    = 0.0
marking  = []
optimizer = optim.Adam(q.parameters(), lr=learning_rate)

num_frames = 15000
batch_size = 32
gamma      = 0.99

losses = []
all_rewards = []
episode_reward = 0

s = env.reset()
for frame_idx in range(1, num_frames + 1):
    a = q.sample_action(s)
    
    s_prime, r, done, _ = env.step(a)
    memory.push(s, a, r, s_prime, done)
    
    s = s_prime
    episode_reward += r
    
    if done:
        s = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        
    if len(memory) > batch_size:
        loss = train(batch_size)
        losses.append(loss.data[0])
        
    #if frame_idx % 200 == 0:
    #    plot(frame_idx, all_rewards, losses)
        
    if frame_idx % 1000 == 0:
        q_target.load_state_dict(q.state_dict())



for n_episode in range(int(1e32)):
    epsilon = max(0.01, 0.08 - 0.01*(n_episode/200)) # linear annealing from 8% to 1%
    s = env.reset()
    done = False
    score = 0.0

    while True:

        a = q.sample_action(torch.from_numpy(s).float().unsqueeze(0))#, epsilon)
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
