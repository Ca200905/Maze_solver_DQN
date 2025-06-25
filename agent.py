import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import collections

# define a named tuple to store a single experience
Transition = collections.namedtuple('Experience',
                                    field_names=['state', 'action',
                                                 'next_state', 'reward',
                                                 'is_game_on'])


class Agent:
    def __init__(self, maze, memory_buffer, use_softmax=True):
        self.env = maze  # environment reference
        self.buffer = memory_buffer  # experience replay buffer
        self.num_act = 4  # number of actions
        self.use_softmax = use_softmax  # whether to use softmax or epsilon-greedy
        self.total_reward = 0  # track cumulative reward
        self.min_reward = -self.env.maze.size  # set minimum to prevent loops
        self.isgameon = True  # track game status

    def make_a_move(self, net, epsilon, device='cuda'):
        # select action using current policy
        action = self.select_action(net, epsilon, device)

        # get current state and perform the action
        current_state = self.env.state()
        next_state, reward, self.isgameon = self.env.state_update(action)
        self.total_reward += reward

        # end game if reward drops too low
        if self.total_reward < self.min_reward:
            self.isgameon = False
        if not self.isgameon:
            self.total_reward = 0

        # store the experience
        transition = Transition(current_state, action,
                                next_state, reward,
                                self.isgameon)
        self.buffer.push(transition)

    def select_action(self, net, epsilon, device='cuda'):
        # convert state to tensor and pass through network
        state = torch.Tensor(self.env.state()).to(device).view(1, -1)
        qvalues = net(state).cpu().detach().numpy().squeeze()

        # sample action using softmax or epsilon-greedy
        if self.use_softmax:
            probs = sp.softmax(qvalues / epsilon).squeeze()
            probs /= np.sum(probs)
            action = np.random.choice(self.num_act, p=probs)
        else:
            if np.random.random() < epsilon:
                action = np.random.randint(self.num_act)
            else:
                action = int(np.argmax(qvalues))

        return action

    def plot_policy_map(self, net, filename, offset):
        # plot policy arrows over the maze
        net.eval()
        with torch.no_grad():
            fig, ax = plt.subplots()
            ax.imshow(self.env.maze, cmap='Greys')

            # compute best action for each valid cell
            for free_cell in self.env.allowed_states:
                self.env.current_position = np.array(free_cell)
                state = torch.Tensor(self.env.state()).view(1, -1).to('cuda')
                qvalues = net(state)
                best_action = int(torch.argmax(qvalues).cpu().numpy())
                arrow = self.env.directions[best_action]

                # draw the arrow at the cell
                ax.text(free_cell[1] - offset[0], free_cell[0] - offset[1], arrow)

            # hide ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # mark the goal in blue
            ax.plot(self.env.goal[1], self.env.goal[0], 'bs', markersize=4)

            # save and show
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()
