import numpy as np, os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical

path = os.getcwd()
abs_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, abs_path + "/utils")
sys.path.insert(0, abs_path + "/env")

import Parameters, Check as ch

# hyperparameters
args = Parameters.parser.parse_args()


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

# For DQN with linear output
class DQN(nn.Module):
    def __init__(self, state_dim, a, device):
        super(DQN, self).__init__()
        self.device = device
        self.state_dim = state_dim
        self.hidden_size = 128
        self.num_layers = 2

        self.gru = nn.GRU(input_size=state_dim,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)

        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)

        self.output = nn.Linear(self.hidden_size, a)
        self.output.weight.data.uniform_(-args.init_w, args.init_w)


    def forward(self, x):
        # Set initial hidden and cell states, c0 for lstm
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.gru(x, h0)
        out = F.relu(self.fc1(out[:, -1, :]))
        out = self.output(out)
        return out


# For DQN with linear output
class DQN2(nn.Module):
    def __init__(self, s, a, device):
        super(DQN2, self).__init__()
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

class DPGCritic(nn.Module):

    def __init__(self, state_dim, a, device):
        super(DPGCritic, self).__init__()
        self.device = device
        self.state_dim = state_dim
        self.hidden_size = 128
        self.num_layers = 2

        self.gru = nn.GRU(input_size=state_dim,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)

        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)

        self.fca1 = nn.Linear(a, self.hidden_size)
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

        self.fc2 = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.output = nn.Linear(self.hidden_size, 1)
        self.output.weight.data.uniform_(-args.init_w, args.init_w)


    def forward(self, x, action):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.gru(x, h0)
        s1 = F.relu(self.fc1(out[:, -1, :]))
        a1 = F.relu(self.fca1(action))
        x = torch.cat((s1, a1), dim=1)
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x


# For DDPG with sigmoid output [0,1]
class DPGActor2(nn.Module):

    def __init__(self, state_dim, a, device):
        super(DPGActor2, self).__init__()
        self.device = device
        self.state_dim = state_dim
        self.hidden_size = 128
        self.num_layers = 2

        self.gru = nn.GRU(input_size=state_dim,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)

        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)

        self.output = nn.Linear(self.hidden_size, a)
        self.output.weight.data.uniform_(-args.init_w, args.init_w)

    def forward(self, x):
        # Set initial hidden and cell states, c0 for lstm
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.gru(x, h0)
        out = F.relu(self.fc1(out[:, -1, :]))
        out = torch.sigmoid(self.output(out))
        return out





# For DQN dueling with linear output
class DQNDueling(nn.Module):
    def __init__(self, s, a):
        super(DQNDueling, self).__init__()
        self.layer1 = nn.Linear(s, args.network1_layer1)
        self.layer1.weight.data = fanin_init(self.layer1.weight.data.size())

        self.layer2 = nn.Linear(args.network1_layer1, args.network1_layer2)
        self.layer2.weight.data = fanin_init(self.layer2.weight.data.size())

        self.advantage = nn.Linear(args.network1_layer2, a)
        self.advantage.weight.data.uniform_(-args.init_w, args.init_w)

        self.value = nn.Linear(args.network1_layer2, 1)
        self.value.weight.data.uniform_(-args.init_w, args.init_w)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()


# For discrete Actor with softmax output
class Policy(nn.Module):
    def __init__(self, state_dim, a, device):
        super(Policy, self).__init__()
        self.device = device
        self.state_dim = state_dim
        self.hidden_size = 128
        self.num_layers = 2

        self.gru = nn.GRU(input_size=state_dim,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)

        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)

        self.output = nn.Linear(self.hidden_size, a)
        self.output.weight.data.uniform_(-args.init_w, args.init_w)


    def forward(self, x):
        # Set initial hidden and cell states, c0 for lstm
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.gru(x, h0)
        out = F.relu(self.fc1(out[:, -1, :]))
        out = F.softmax(self.output(out), dim=-1)
        return out


# For continuous Actor with linear output (to generate parameter for Normal distribution)
class ContinuePolicy(nn.Module):
    def __init__(self, s, a, action_lim):
        self.action_lim = action_lim
        super(ContinuePolicy, self).__init__()
        self.layer1 = nn.Linear(s, args.network1_layer1)
        self.layer1.weight.data = fanin_init(self.layer1.weight.data.size())

        self.layer2 = nn.Linear(args.network1_layer1, args.network1_layer2)
        self.layer2.weight.data = fanin_init(self.layer2.weight.data.size())

        self.final_mean = nn.Linear(args.network1_layer2, a)
        self.final_mean.weight.data.uniform_(-args.init_w, args.init_w)
        self.final_mean.bias.data.uniform_(-args.init_w, args.init_w)

        self.final_logstd = nn.Linear(args.network1_layer2, a)
        self.final_logstd.weight.data.uniform_(-args.init_w, args.init_w)
        self.final_logstd.bias.data.uniform_(-args.init_w, args.init_w)

        # self.log_std = nn.Parameter(torch.ones(1, a) * 0.0)

    def forward(self, x):
        x = F.tanh(self.layer1(x))
        x = F.tanh(self.layer2(x))
        mean = self.final_mean(x)/10
        std = F.softplus(self.final_logstd(x))/10
        # std = self.log_std.exp().expand_as(mean)/10
        dist = Normal(mean, std)
        return dist
        # mean = self.final_mean(x)
        # log_std = self.final_logstd(x)
        # log_std = torch.clamp(log_std, -0.5+1e-5, 0.5)
        # std = log_std.exp()
        # return mean, std, log_std



    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mean, log_std


# For critic in AC with linear output
class Critic(nn.Module):
    def __init__(self, state_dim, a, device):
        super(Critic, self).__init__()
        self.device = device
        self.state_dim = state_dim
        self.hidden_size = 128
        self.num_layers = 2

        self.gru = nn.GRU(input_size=state_dim,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)

        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)

        # self.fca1 = nn.Linear(1, self.hidden_size)
        # self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.output = nn.Linear(self.hidden_size, 1)
        self.output.weight.data.uniform_(-args.init_w, args.init_w)


    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.gru(x, h0)
        x = F.relu(self.fc1(out[:, -1, :]))
        # a1 = F.relu(self.fca1(action))
        # x = torch.cat((s1, a1), dim=1)
        x = F.relu(self.fc2(x))
        x = self.output(x)
        
        return x


# For combined AC Actor with softmax output (actor) and linear output (critic)
class ActorCritic(nn.Module):
    def __init__(self, s, a):
        super(ActorCritic, self).__init__()
        self.layer1 = nn.Linear(s, args.network1_layer1)
        self.layer1.weight.data = fanin_init(self.layer1.weight.data.size())

        self.layer2 = nn.Linear(args.network1_layer1, args.network1_layer2)
        self.layer2.weight.data = fanin_init(self.layer2.weight.data.size())

        self.actor = nn.Linear(args.network1_layer2, a)
        self.actor.weight.data.uniform_(-args.init_w, args.init_w)

        self.critic = nn.Linear(args.network1_layer2, 1)
        self.critic.weight.data.uniform_(-args.init_w, args.init_w)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        actor = F.softmax(self.actor(x), dim=-1)
        critic = self.critic(x)
        return actor, critic

# # # For traffic prediction
# class predictor_LSTM(nn.Module):

#     def __init__(self, state_dim, predict_l, traffic_up, device):
#         super(predictor_LSTM, self).__init__()
#         self.device = device
#         self.state_dim = state_dim
#         self.hidden_size = args.network1_layer1
#         self.predict_l = predict_l

#         self.lstm1 = nn.LSTMCell(state_dim, args.network1_layer1)
#         self.lstm2 = nn.LSTMCell(args.network1_layer1, args.network1_layer1)

#         self.output = nn.Linear(args.network1_layer1, traffic_up)
#         self.output.weight.data.uniform_(-args.init_w, args.init_w)

#     def forward(self, input):
#         future = self.predict_l
#         outputs = []
#         h_t = torch.zeros(input.size(0), args.network1_layer1, dtype=torch.float).to(self.device)
#         c_t = torch.zeros(input.size(0), args.network1_layer1, dtype=torch.float).to(self.device)
#         h_t2 = torch.zeros(input.size(0), args.network1_layer1, dtype=torch.float).to(self.device)
#         c_t2 = torch.zeros(input.size(0), args.network1_layer1, dtype=torch.float).to(self.device)
#         # ch.S(np.shape(h_t.data.numpy()))

#         for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
#             # len(input_t) is the number of samples in this batch
#             input_t = torch.reshape(input_t, [len(input_t), self.state_dim])
#             h_t, c_t = self.lstm1(input_t, (h_t, c_t))
#             h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
#             output = F.softmax(self.output(h_t2), dim=-1)
#             outputs += [output]
#         for i in range(future):
#             # if we should predict the future
#             no_input = torch.zeros_like(input_t)
#             h_t, c_t = self.lstm1(no_input, (h_t, c_t))
#             h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
#             output = F.softmax(self.output(h_t2), dim=-1)
#             outputs += [output]
#         outputs = torch.stack(outputs, 1).squeeze(2)
#         return outputs


# # For traffic prediction
# class predictor_GRU(nn.Module):

#     def __init__(self, state_dim, predict_l, input_l, traffic_up, device):
#         super(predictor_GRU, self).__init__()
#         self.device = device
#         self.state_dim = state_dim
#         self.hidden_size = 128
#         self.predict_l = predict_l+input_l
#         self.num_layers = 2

#         self.gru = nn.GRU(input_size=state_dim,
#                           hidden_size=self.hidden_size,
#                           num_layers=self.num_layers)


#         self.output = nn.Linear(self.hidden_size, traffic_up)
#         self.output.weight.data.uniform_(-args.init_w, args.init_w)

#     def init_hidden(self, batch_size):
#         return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
#         # torch.autograd.Variable(torch.randn(self.num_layers, batch_size, self.hidden_size, device=self.device))

#     def forward(self, input, hidden):
#         x, hidden = self.gru(input, hidden)
#         output = F.softmax(self.output(x), dim=2)
#         return output, hidden


# # For DDPG with tanh output [-1, 1]
# class DPGActor(nn.Module):

#     def __init__(self, state_dim, action_dim, action_lim):
#         super(DPGActor, self).__init__()
#         self.action_lim = action_lim

#         self.fc1 = nn.Linear(state_dim, args.network1_layer1)
#         self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

#         self.fc2 = nn.Linear(args.network1_layer1, args.network1_layer2)
#         self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

#         self.fc3 = nn.Linear(args.network1_layer2, action_dim)
#         self.fc3.weight.data.uniform_(-args.init_w, args.init_w)

#     def forward(self, state):
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         action = torch.tanh(self.fc3(x))
#         action = action * self.action_lim
#         return action

# For DDPG with linear output




