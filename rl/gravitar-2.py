# this is a Deep Q Learning (DQN) agent including replay memory and a target network 
# you can write a brief 8-10 line abstract detailing your submission and experiments here
# the code is based on https://github.com/seungeunrho/minimalRL/blob/master/dqn.py, which is released under the MIT licesne
# make sure you reference any code you have studied as above, with one comment line per reference

# imports
import math
import gym
import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32
video_every   = 25
print_every   = 5

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, s, a, r, s_prime, done):
        s = np.expand_dims(s, 0)
        s_prime = np.expand_dims(s_prime, 0)
        self.buffer.append((s,a,r,s_prime,done))
    
    def sample(self, n):
        s,a,r,s_prime,done = zip(*random.sample(self.buffer, n))
        return np.concatenate(s),a,r,np.concatenate(s_prime),done
        #mini_batch = random.sample(self.buffer, n)
        #s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        #for transition in mini_batch:
        #    s, a, r, s_prime, done_mask = transition
        #    s_lst.append(s)
        #    a_lst.append([a])
        #    r_lst.append([r])
        #    s_prime_lst.append(s_prime)
        #    done_mask_lst.append([done_mask])

        #return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
        #       torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
        #       torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(QNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n)
        )
        #self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        #self.fc2 = nn.Linear(256, 84)
        #self.fc3 = nn.Linear(84, env.action_space.n)

    def forward(self, x):
        return self.layers(x)
        #x = x.view(x.size(0),-1)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        #return x
      
    def sample_action(self, s, epsilon):
        if random.random() > epsilon:
            s = torch.FloatTensor(s).unsqueeze(0)
            q_value = self.forward(s)
            a = q_value.max(1)[1].data[0]
        else:
            a = random.randrange(env.action_space.n)
        return a
            
def train(q, q_target, memory, optimizer):
    #for i in range(10):
    s,a,r,s_prime,done_mask = memory.sample(batch_size)

    s = torch.FloatTensor(np.float32(s))
    s_prime = torch.FloatTensor(np.float32(s))
    a = torch.LongTensor(a)
    r = torch.FloatTensor(r)
    done_mask = torch.FloatTensor(done_mask)

    q_values = q(s)
    next_q_values = q(s_prime)

    q_value          = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
    max_q_value     = next_q_values.max(1)[0]
    expected_q_value = r + gamma * max_q_value * done_mask
    
    loss = (q_value - expected_q_value.data).pow(2).mean()

    #q_out = q(s)
    #q_a = q_out.gather(1,a)
    #max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
    #target = r + gamma * max_q_prime * done_mask
    #loss = F.smooth_l1_loss(q_a, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

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

q = QNetwork(env.observation_space.shape[0], env.action_space.n)
q_target = QNetwork(env.observation_space.shape[0], env.action_space.n)
q_target.load_state_dict(q.state_dict())
memory = ReplayBuffer()

score    = 0.0
marking  = []
optimizer = optim.Adam(q.parameters(), lr=learning_rate)

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500
epsilon_by_frame = lambda n_episode: epsilon_final + (epsilon_start-epsilon_final)*math.exp(-1.*n_episode/epsilon_decay)
for n_episode in range(int(1e32)):
    epsilon = epsilon_by_frame(n_episode) # epsilon greedy
    #epsilon = max(0.01, 0.08 - 0.01*(n_episode/200)) # linear annealing from 8% to 1%
    s = env.reset()
    done = False
    score = 0.0

    while True:

        #a = q.sample_action(torch.from_numpy(s).float().unsqueeze(0), epsilon)
        a = q.sample_action(s,epsilon)
        s_prime, r, done, info = env.step(a)
        done_mask = 0.0 if done else 1.0
        memory.put(s,a,r/100.0,s_prime, done_mask)
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
