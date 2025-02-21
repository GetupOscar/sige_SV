# HyperParameters for Learning Agents
import argparse

parser = argparse.ArgumentParser(description='PyTorch actor-critic')


parser.add_argument('--lr_rnn', type=float, default=0.0001, metavar='N1',
                    help='learning rate for rnn (default: 1e-4)')
parser.add_argument('--lr_a', type=float, default=1e-5, metavar='N1',
                    help='learning rate for actor (default: 1e-4)')
parser.add_argument('--lr_c', type=float, default=1e-4, metavar='N1',
                    help='learning rate for critic (default: 1e-3)')
parser.add_argument('--network1_layer1', type=int, default=128, metavar='N1',
                    help='layer1 for N1')
parser.add_argument('--network1_layer2', type=int, default=128, metavar='N1',
                    help='layer1 for N1')
parser.add_argument('--tau', type=float, default=0.001, metavar='N1',
                    help='target network update rate')
parser.add_argument('--init_w', type=float, default=0.003, metavar='N1',
                    help='init_w for N1')

parser.add_argument('--gamma', type=float, default=0.9, metavar='N1',
                    help='discount factor (default: 0.99)')
parser.add_argument('--gamma_acb', type=float, default=0.1, metavar='N1',
                    help='discount factor (default: 0.99)')
parser.add_argument('--update_frequency', type=int, default=1000, metavar='N1',
                    help='network update frequency')
parser.add_argument('--memory_capacity', type=int, default=10000, metavar='N1',
                    help='memory capacity of training samples')
parser.add_argument('--random_batch_size', type=int, default=32, metavar='N1',
                    help='size of minibatch')
parser.add_argument('--batch_size', type=int, default=32, metavar='N1',
                    help='size of minibatch')

parser.add_argument('--max_epsilon', type=float, default=1, metavar='N1',
                    help='maximum epsilon, 1 refers to completely random action choosing')
parser.add_argument('--min_epsilon', type=float, default=0.01, metavar='N1',
                    help='minimal epsilon for exploration')
parser.add_argument('--epsilon_decay_rate', type=float, default=0.0001, metavar='N1',
                    help='epsilon_decay_rate')
