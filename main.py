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
from tqdm import tqdm
from ShapleyExplainer import ShapleyExplainer

np.random.seed(1)
LR = 0.2 #learning rate
GAMMA = 0.9  # discount factor
MAX_EPISODES = 50  # maximum episodes
network = VR_environment.Network()
iteration = np.zeros(MAX_EPISODES, dtype=int)
B = 100
TARGET_REPLACE_ITER = 20

args = Parameters.parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize arrays for metrics
dqn_loss = np.zeros(MAX_EPISODES, dtype=float)
qoe = np.zeros(MAX_EPISODES, dtype=int)
running_qoe = np.zeros(MAX_EPISODES, dtype=float)
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
        self.state_dim = s
        self.action_dim = a
        
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
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = Memories.RandomMemory()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.number = MAX_EPISODES
        self.epsilon = 0.9
        self.steps = 0
        self.current_action_dim = a  # Track current action dimension
        
        # Initialize Shapley explainer with enhanced logging
        self.shapley_explainer = ShapleyExplainer(self.eval_net)

    def choose_action(self, s, action_dim, perms_length):
        """
        Choose action with additional safety checks
        perms_length: actual length of permutations list
        """
        # Update current action dimension
        self.current_action_dim = perms_length
        
        if random.random() > self.epsilon:
            action = random.randint(0, perms_length - 1)
        else:
            s = np.array(s).reshape(1, -1)
            pr_a = Variable(torch.FloatTensor(s).to(device)).detach()
            action_value = self.eval_net.forward(pr_a)
            # Ensure action is within current permutations range
            action = torch.clamp(torch.max(action_value, 1)[1], 0, perms_length - 1).data[0].numpy()
        
        return int(action)

    def learn(self):
        if self.learn_step_counter % 5 == 0 and len(self.memory.samples) > 0:
            b_s, _, _, _ = self.memory.sample(1)
            b_s = b_s.reshape(1, -1)
            state_tensor = torch.FloatTensor(b_s).to(device)
            self.shapley_explainer.explain(state_tensor)
            
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
                
        self.learn_step_counter += 1
        
        try:
            b_s, b_a, b_r, b_s_ = self.memory.sample()
            
            # Ensure actions are within current range
            b_a = np.clip(b_a, 0, self.current_action_dim - 1)
            
            b_s = Variable(torch.from_numpy(b_s)).to(device)
            b_a = Variable(torch.from_numpy(b_a).type(torch.LongTensor)).to('cpu')
            b_r = Variable(torch.from_numpy(b_r)).to('cpu')
            b_s_ = Variable(torch.from_numpy(b_s_)).to(device)

            q_eval = self.eval_net(b_s).type('torch.FloatTensor').gather(1, b_a)
            q_next = self.target_net(b_s_).to('cpu').detach()
            
            # Ensure next actions are within current range
            next_actions = torch.clamp(torch.max(q_next, 1)[1], 0, self.current_action_dim - 1)
            q_next = self.target_net(b_s_).type('torch.FloatTensor').gather(1, next_actions.unsqueeze(1)).detach()
            q_next = torch.squeeze(q_next)
            q_target = b_r + q_next * args.gamma_acb

            loss = F.mse_loss(q_eval, q_target.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.eval_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            self.steps += 1
            return loss
        except Exception as e:
            print(f"Warning: Learning step failed - {str(e)}")
            return torch.tensor(0.0)

network.VR_spherical_rendering()
network.VR_requirement()

# Create progress bar for training
progress_bar = tqdm(total=MAX_EPISODES, desc='Training Progress')
episode = 0

while (episode < MAX_EPISODES):
    # Initialize actions and get permutations
    ACTIONS = list(range(network.mec))
    
    # Debug FOV index state
    if episode % 10 == 0:
        print(f"\nFOV Analysis:")
        print(f"FOV indices: {network.fov_index}")
        print(f"FOV index length: {len(network.fov_index)}")
    
    # Generate permutations safely
    try:
        perms = list(permutations(range(len(ACTIONS)), len(network.fov_index)))
        perms_length = len(perms)
        if perms_length == 0:
            print("Warning: No valid permutations generated")
            perms = [(0,)]  # Fallback to simple action
            perms_length = 1
    except Exception as e:
        print(f"Error generating permutations: {e}")
        perms = [(0,)]  # Fallback to simple action
        perms_length = 1
    
    if episode == 0:
        state_dim = 5
        action_dim = perms_length
        dqn = DQN(state_dim, action_dim)
    else:
        # Update network output dimension if needed
        if perms_length != dqn.current_action_dim:
            print(f"\nAction space changed: {dqn.current_action_dim} -> {perms_length}")
            dqn.eval_net.final = nn.Linear(args.network1_layer2, perms_length).to(device)
            dqn.target_net.final = nn.Linear(args.network1_layer2, perms_length).to(device)
            dqn.current_action_dim = perms_length
    
    # Create state vector with all features
    current_state = np.array([
        np.mean(network.user_mec_distance),
        np.mean(network.MEC_process_ability),
        np.mean(network.user_x),
        np.mean(network.user_y),
        np.mean(network.R_down)
    ])
    
    # Debug state information
    # if episode % 10 == 0:
    #     print(f"\nState Analysis (Episode {episode}):")
    #     print(f"User-MEC Distance: {current_state[0]:.2f}")
    #     print(f"MEC Process Ability: {current_state[1]:.2f}")
    #     print(f"User Position (x,y): ({current_state[2]:.2f}, {current_state[3]:.2f})")
    #     print(f"Bandwidth: {current_state[4]:.2f}")
    #     print(f"\nAction Space:")
    #     print(f"Number of MECs: {network.mec}")
    #     print(f"Number of permutations: {perms_length}")
    
    # Get and validate action
    action_index = dqn.choose_action(current_state, action_dim, perms_length)
    if episode % 10 == 0:
        print(f"Selected action index: {action_index}")
    
    select_action = perms[action_index]
    
    # Execute action and get immediate results
    network.downlink_transmission_for_q_learning_new(select_action, network.fov_index)
    R_down = network.R_down
    R_down_new = R_down * B
    
    # Calculate render times and delays
    downlink_time_VR_device = np.zeros(network.user, dtype=float)
    render_time_VR_device = np.zeros(network.user, dtype=float)
    downlink_time = np.zeros(network.user, dtype=float)
    render_time = np.zeros(network.user, dtype=float)
    
    for i in range(len(select_action)):
        for j in range(len(network.fov_index[i])):
            render_time[network.fov_index[i][j]] = network.fov_frame_cycle * 1000 / (network.MEC_process_ability[select_action[i]] * network.GPU * network.thread * 1000000000)
            render_time_VR_device[network.fov_index[i][j]] = network.original_frame_cycle * 1000 / (network.VR_process_ability * network.GPU * network.thread * 1000000000)
    
    for i in range(network.user):
        # Avoid division by zero
        if R_down_new[i] > 0:
            downlink_time[i] = network.fov_frame * 1000 / R_down_new[i]
            downlink_time_VR_device[i] = network.original_frame * 1000 / R_down_new[i]
        else:
            downlink_time[i] = float('inf')  # Set to infinity for zero bandwidth
            downlink_time_VR_device[i] = float('inf')
    
    # Calculate means
    downlink_time_mean[episode] = np.mean(downlink_time)
    render_time_mean[episode] = np.mean(render_time)
    downlink_time_VR_device_mean[episode] = np.mean(downlink_time_VR_device)
    render_time_VR_device_mean[episode] = np.mean(render_time_VR_device)
    total_time_mean[episode] = downlink_time_mean[episode] + render_time_mean[episode]
    total_time_VR_device_mean[episode] = downlink_time_VR_device_mean[episode] + render_time_VR_device_mean[episode]
    
    # Calculate QoE
    network.QoE()
    qoe[episode] = network.sum_V_PSNR
    running_qoe[episode] = (1 - GAMMA) * network.sum_V_PSNR + GAMMA * running_qoe[episode - 1]
    
    # Update environment state
    network.user_mobility()
    network.calculate_mec_user_distance()
    network.calculate_user_mec_distance()
    network.VR_requirement()
    
    # Get next state after all updates
    next_state = np.array([
        np.mean(network.user_mec_distance),  # 距离会随用户移动变化
        np.mean(network.MEC_process_ability),  # 计算能力保持不变
        np.mean(network.user_x),  # 用户X坐标会随移动变化
        np.mean(network.user_y),  # 用户Y坐标会随移动变化
        np.mean(network.R_down)  # 带宽会随距离变化而更新
    ])
    
    # Debug state transition details
    # if episode % 10 == 0:
    #     state_changes = next_state - current_state
    #     print("\nState Transition Analysis:")
    #     print(f"Distance Change: {state_changes[0]:.2f}")
    #     print(f"Compute Change: {state_changes[1]:.2f}")
    #     print(f"Position X Change: {state_changes[2]:.2f}")
    #     print(f"Position Y Change: {state_changes[3]:.2f}")
    #     print(f"Bandwidth Change: {state_changes[4]:.2f}")
        
    #     print("\nUser Movement Analysis:")
    #     print(f"Average X Position: {np.mean(network.user_x):.2f}")
    #     print(f"Average Y Position: {np.mean(network.user_y):.2f}")
    #     print(f"X Position Std: {np.std(network.user_x):.2f}")
    #     print(f"Y Position Std: {np.std(network.user_y):.2f}")
    
    # Validate state changes
    if np.any(np.isnan(next_state)):
        print("Warning: NaN values detected in next_state")
        next_state = np.nan_to_num(next_state, 0.0)
    
    if np.any(np.isinf(next_state)):
        print("Warning: Inf values detected in next_state")
        next_state = np.clip(next_state, -1e6, 1e6)
    
    # Store transition with validated states
    dqn.memory.add(current_state, [action_index], running_qoe[episode], next_state)
    
    # Learn and update progress
    dqn_loss[episode] = dqn.learn()
    current_state = next_state.copy()
    iteration[episode] = episode
    
    # Update progress bar with current metrics
    progress_bar.set_postfix({
        'QoE': f'{running_qoe[episode]:.2f}',
        'Loss': f'{dqn_loss[episode]:.4f}',
        'Epsilon': f'{dqn.epsilon:.2f}'
    })
    progress_bar.update(1)
    
    # Save Shapley analysis periodically
    if episode % 10 == 0:
        dqn.shapley_explainer.save_history()
    
    episode += 1

progress_bar.close()

# Final Shapley analysis and visualization
print("\nGenerating Shapley value analysis...")
dqn.shapley_explainer.save_history()

# Create output directory if it doesn't exist
import os
output_dir = os.path.join(os.getcwd(), 'results')
os.makedirs(output_dir, exist_ok=True)

timestamp = time.strftime("%Y%m%d_%H%M%S")

print(f"\nSaving results to: {output_dir}")

# Plot feature importance without error bars
print("Generating basic Shapley plot...")
dqn.shapley_explainer.plot_summary(with_error=False)
basic_plot_path = os.path.join(output_dir, f'shapley_basic_{timestamp}.png')
plt.savefig(basic_plot_path)


# Plot feature importance with error bars
print("Generating Shapley plot with error bars...")
dqn.shapley_explainer.plot_summary(with_error=True)
error_plot_path = os.path.join(output_dir, f'shapley_with_error_{timestamp}.png')
plt.savefig(error_plot_path)


# Plot training metrics
print("Generating training metrics plot...")
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(iteration, running_qoe)
plt.xlabel('Episode')
plt.ylabel('QoE')
plt.title('Running QoE vs Episode')

plt.subplot(1, 2, 2)
plt.plot(iteration, total_time_mean, color='green', label='Render in MEC')
plt.plot(iteration, total_time_VR_device_mean, color='red', label='Render in VR Device')
plt.xlabel('Episode')
plt.ylabel('Time (ms)')
plt.title('Rendering Delay Comparison')
plt.legend()

plt.tight_layout()
metrics_plot_path = os.path.join(output_dir, f'training_metrics_{timestamp}.png')
plt.savefig(metrics_plot_path)


# Save Shapley analysis data
print("Saving Shapley analysis data...")
analysis_path = os.path.join(output_dir, f'shapley_analysis_{timestamp}.json')
dqn.shapley_explainer.save_history(analysis_path)
print(f"Saved to: {analysis_path}")

print("\nTraining complete! All results saved to:")
print(f"Directory: {output_dir}")
# print("Files:")
# print(f"- shapley_basic_{timestamp}.png")
# print(f"- shapley_with_error_{timestamp}.png")
# print(f"- training_metrics_{timestamp}.png")
# print(f"- shapley_analysis_{timestamp}.json")
