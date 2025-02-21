import numpy as np
from itertools import combinations, permutations
import pandas as pd
import time
import VR_environment
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import Memories
import Parameters
import math
import utils
import heapq
#from NetworkModel import DQN
# import DQN_ACB

np.random.seed(1)
# tf.set_random_seed(1)
LR = 0.2 #learning rate
#BATCH_SIZE = 32
GAMMA = 0.9  # discount factor
MAX_EPISODES = 200  # maximum episodes
network = VR_environment.Network()
iteration = np.zeros(MAX_EPISODES, dtype=int)
B = 100
TARGET_REPLACE_ITER = 20

args = Parameters.parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dqn_loss = np.zeros(MAX_EPISODES, dtype=float)
dqn_target = np.zeros(MAX_EPISODES, dtype=float)
qoe = np.zeros(MAX_EPISODES, dtype=int)
running_reward = np.zeros(MAX_EPISODES, dtype=float)
episode = 0
running_qoe = np.zeros(MAX_EPISODES, dtype=float)
render_delay = np.zeros(MAX_EPISODES, dtype=float)
downlink_time_mean = np.zeros(MAX_EPISODES, dtype=float)
render_time_mean = np.zeros(MAX_EPISODES, dtype=float)
total_time_mean = np.zeros(MAX_EPISODES, dtype=float)
downlink_time_VR_device_mean = np.zeros(MAX_EPISODES, dtype=float)
render_time_VR_device_mean = np.zeros(MAX_EPISODES, dtype=float)
total_time_VR_device_mean = np.zeros(MAX_EPISODES, dtype=float)



def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Net(nn.Module):
    def __init__(self, s, a):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(s, args.network1_layer1)
        self.layer1.weight.data = fanin_init(self.layer1.weight.data.size())

        self.layer2 = nn.Linear(args.network1_layer1, args.network1_layer2)
        self.layer2.weight.data = fanin_init(self.layer2.weight.data.size())

        self.final = nn.Linear(args.network1_layer2, a)
        self.final.weight.data.uniform_(-args.init_w, args.init_w)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.final(x)
        return x

class DQN(object):
    def __init__(self, s, a):
        self.eval_net, self.target_net = Net(s, a), Net(s, a)

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = Memories.RandomMemory()  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.number = MAX_EPISODES
        #self.epsilon = args.max_epsilon  # greedy police
        self.epsilon = 0.9
        self.steps = 0

    def choose_action(self, s, action_dim):
        if random.random() > self.epsilon:
            return random.randint(0, action_dim - 1)
        else:
            pr_a = Variable(torch.FloatTensor([s]).to(device)).detach()
            action_value = self.eval_net.forward(pr_a)
            #print('action_value: ', action_value)
            #print(torch.max(action_value, 1))
            action = torch.max(action_value, 1)[1].data[0].numpy()
            #action = action[0]#pr_a.max(1)[1].to('cpu').data[0].numpy()
            #print('action: ', action)
        return action

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # target parameter update
        b_s, b_a, b_r, b_s_ = self.memory.sample()
        #print('b_s_: ', b_s_)
        b_s = Variable(torch.from_numpy(b_s)).to(device)
        b_a = Variable(torch.from_numpy(b_a).type(torch.LongTensor)).to('cpu')
        b_r = Variable(torch.from_numpy(b_r)).to('cpu')
        b_s_ = Variable(torch.from_numpy(b_s_)).to(device)
        #print('b_s_:', b_s_)

        # q_eval = self.eval_net(b_s)
        q_eval = self.eval_net(b_s).type('torch.FloatTensor').gather(1, b_a)
        q_eval = torch.squeeze(q_eval)
        #print('q_eval: ', q_eval)
        # print('b_s_: ', b_s_)
        q_next = self.target_net(b_s_).to('cpu').detach()
        #print('q_next: ', q_next)
        q_next = self.target_net(b_s_).type('torch.FloatTensor').gather(1, torch.max(q_next, 1)[1].unsqueeze(1)).detach()
        #print('q_next1: ', q_next)
        q_next = torch.squeeze(q_next)
        #print('q_next: ', q_next)
        #print('args.gamma_acb: ', args.gamma_acb)
        q_target = b_r + q_next * args.gamma_acb
        #print('q_target:', q_target)


        #print('q_eval: ', q_eval)
        #print('q_target: ', q_target)

        loss = F.mse_loss(q_eval, q_target)
        self.optimizer.zero_grad()
        #print('loss: ', loss)
        loss.backward()
        for param in self.eval_net.parameters():
            param.grad.data.clamp_(-1, 1)
        #target = sum(q_target)
        # print('target: ', target)
        #target = target.detach().numpy().flatten()
        #self.optimizer.zero_grad()
        #loss.backward()
        self.optimizer.step()
        self.steps += 1
        #self.epsilon = args.min_epsilon + (args.max_epsilon - args.min_epsilon) * math.exp(-args.epsilon_decay_rate * self.steps)
        #print('epsilon: ', self.epsilon)
        #return loss, target.max()
        return loss

network.VR_spherical_rendering()
network.VR_requirement()
while (episode < MAX_EPISODES):

    network.calculate_mec_user_distance()
    network.calculate_user_mec_distance()
    ACTIONS = []
    for i in range(network.mec):
        ACTIONS.append(i)
    perms = permutations(range(len(ACTIONS)), len(network.fov_index))
    perms = list(perms)
    # print('perms: ', perms)
    network.user_mobility()

    #state_dim = len(network.fov_index)
    state_dim = 1
    action_dim = len(perms)

    dqn = DQN(state_dim, action_dim)
    action_index = int(dqn.choose_action([state_dim], action_dim))
    #print('action_index: ', action_index)
    select_action = perms[action_index]
    network.downlink_transmission_for_q_learning_new(select_action,network.fov_index)
    R_down = network.R_down
    R_down_new = R_down * B
    #print('R_down_new: ', R_down_new)
    downlink_time_VR_device = np.zeros(network.user, dtype=float)
    render_time_VR_device = np.zeros(network.user, dtype=float)
    downlink_time = np.zeros(network.user, dtype=float)
    render_time = np.zeros(network.user, dtype=float)
    for i in range(len(select_action)):
        for j in range(len(network.fov_index[i])):
            render_time[network.fov_index[i][j]] = network.fov_frame_cycle * 1000 / (network.MEC_process_ability[select_action[i]] * network.GPU * network.thread * 1000000000)
            render_time_VR_device[network.fov_index[i][j]] = network.original_frame_cycle * 1000 / (network.VR_process_ability * network.GPU * network.thread * 1000000000)
    for i in range(network.user):
        downlink_time[i] = network.fov_frame * 1000 / R_down_new[i]
        downlink_time_VR_device[i] = network.original_frame * 1000 / R_down_new[i]
    downlink_time_sum = sum(downlink_time)
    render_time_sum = sum(render_time)
    downlink_time_VR_device_sum = sum(downlink_time_VR_device)
    render_time_VR_device_sum = sum(render_time_VR_device)
    #print('downlink_time_sum: ', downlink_time_sum/network.user)
    downlink_time_mean[episode] = downlink_time_sum/network.user
    render_time_mean[episode] = render_time_sum / network.user
    downlink_time_VR_device_mean[episode] = downlink_time_VR_device_sum / network.user
    render_time_VR_device_mean[episode] = render_time_VR_device_sum / network.user
    total_time_mean[episode] = downlink_time_mean[episode] + render_time_mean[episode]
    total_time_VR_device_mean[episode] = downlink_time_VR_device_mean[episode] + render_time_VR_device_mean[episode]
    network.QoE()
    qoe[episode] = network.sum_V_PSNR
    running_qoe[episode] = (1 - GAMMA) * network.sum_V_PSNR + GAMMA * running_qoe[episode - 1]
    dqn.memory.add([state_dim], [action_index], running_qoe[episode], [state_dim])
    dqn_loss[episode] = dqn.learn()

    #print('select_action: ', select_action)
    iteration[episode] = episode
    # print(episode)
    episode = episode + 1

# print(running_qoe)
# plt.plot(iteration, running_qoe)
# plt.show()

# # print(total_time_mean)
# # print(total_time_VR_device_mean)
# plt.plot(iteration, total_time_mean, color='green', label='Render in MEC')
# plt.plot(iteration, total_time_VR_device_mean, color='red', label='Render in VR Device')
# plt.legend()
# plt.xlabel('Epoch')
# plt.ylabel('Delay (ms)')
# plt.show()

import matplotlib.pyplot as plt

# 第一张图：QoE 随迭代变化
plt.figure()
plt.plot(iteration, running_qoe)
plt.xlabel('Epoch')
plt.ylabel('QoE')
plt.title('QoE vs. Epoch')
plt.draw()  # 立即绘制，但不阻塞

# 第二张图：不同渲染方式的延迟
plt.figure()
plt.plot(iteration, total_time_mean, color='green', label='Render in MEC')
plt.plot(iteration, total_time_VR_device_mean, color='red', label='Render in VR Device')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Delay (ms)')
plt.title('Rendering Delay Comparison')
plt.draw()  # 立即绘制，但不阻塞

plt.show()  # 让所有图像同时显示


#print(running_user_qoe)
#plt.plot(iteration, running_user_qoe)
#plt.show()


'''
print(downlink_time_mean)
plt.plot(iteration, downlink_time_mean)
plt.show()
print(running_reward)
plt.plot(iteration, running_reward)
plt.show()

print(total_reward)
plt.plot(iteration, total_reward)
plt.show()

print(qoe_average)
plt.plot(iteration, qoe_average)
plt.show()

print(render_running_reward)
plt.plot(iteration, render_running_reward)
plt.show()

print(dqn_loss)
plt.plot(iteration, dqn_loss)
plt.show()

print(uplink_transmission_time_mean)
plt.plot(iteration, uplink_transmission_time_mean)
plt.show()

print(total_time)
plt.plot(iteration, total_time)
plt.show()

user_render_time_mean = user_render_time_mean/10
print(user_render_time_mean)
plt.plot(iteration, user_render_time_mean)
plt.show()


#downlink_time_mean = list(downlink_time_mean)
#downlink_time_mean.sort(reverse=True)
print(downlink_time_mean)
#print(downlink_time_mean_new)
plt.plot(iteration, downlink_time_mean)
plt.show()


print(running_reward)
plt.plot(iteration, running_reward)
plt.show()

print(dqn_loss)
plt.plot(iteration, dqn_loss)
plt.show()

print(render_running_reward)
plt.plot(iteration, render_running_reward)
plt.show()
# print(dqn_target)
# plt.plot(iteration, dqn_target)
# plt.show()

print(qoe)
plt.plot(iteration, qoe)
plt.show()

print('count_random_render_selection: ', count_random_render_selection)
print('count: ', count)
'''
