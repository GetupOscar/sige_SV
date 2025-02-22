import random, os, sys
import numpy as np
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch.autograd import Variable

path = os.getcwd()
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()  # 如果 __file__ 不存在，使用当前工作目录

abs_path = os.path.abspath(os.path.join(script_dir, ".."))
print(abs_path)

sys.path.insert(0, abs_path + "/utils")
sys.path.insert(0, abs_path + "/env")

import Parameters
from typing import Dict, List

# hyper parameters
args = Parameters.parser.parse_args()

# Discounted current sample for on-Policy algorithm
class GeneralMemory:
    samples = []
    # tuple structure
    Transition = namedtuple("Transition", field_names=["state", "action", "reward", "next_state"])


    def add(self, *samples):
        self.samples.append(self.Transition(*samples))
        #print('len(self.samples): ', len(self.samples))
        # delete old sample if memory capacity is full
        if len(self.samples) > args.memory_capacity:
            self.samples.pop(0)
        #print('sample: ', samples)

    def sample(self):
        n = min(args.batch_size, len(self.samples))
        #print('n: ', n)
        batch = self.samples[len(self.samples)-n:len(self.samples)]
        for i in range(n):
            self.samples.pop(-1)
        #print('batch: ', batch)

        s_arr = np.flipud(np.float32([i.state for i in batch]))
        a_arr = np.flipud(np.float32([i.action for i in batch]))
        r_arr = np.flipud(np.float32([i.reward for i in batch]))
        ss_arr = np.flipud(np.float32([i.next_state for i in batch]))
        #print('s_arr: ', s_arr)
        #print('a_arr: ', a_arr)

        return s_arr, a_arr, r_arr, ss_arr

# Random sampled memory for DQN,DDPG
class RandomMemory:
    samples = []
    # tuple structure
    Transition = namedtuple("Transition", field_names=["state", "action", "reward", "next_state"])

    def add(self, *samples):
        self.samples.append(self.Transition(*samples))
        # delete old sample if memory capacity is full
        if len(self.samples) > args.memory_capacity:
            self.samples.pop(0)

    def sample(self, n=None):
        if n is None:
            n = min(args.random_batch_size, len(self.samples))
        else:
            n = min(n, len(self.samples))
            
        batch = random.sample(self.samples, n)

        s_arr = np.float32([i.state for i in batch])
        a_arr = np.float32([i.action for i in batch])
        r_arr = np.float32([i.reward for i in batch])
        s1_arr = np.float32([i.next_state for i in batch])

        return s_arr, a_arr, r_arr, s1_arr

'''
memory = RandomMemory()
for i in range(10):
    memory.add( [1, 2], i, i, i)
print(memory.samples)
s_arr, a1, r_arr, s1_arr = memory.sample()
#print(len(a1))
a1 = Variable(torch.from_numpy(a1).type(torch.LongTensor)).to('cpu')
#print(a1)
#a1 = a1.T[0]
#a1 = np.reshape(a1, [len(a1), 1])
#print(a1)

#x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
'''
