import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import copy

class MazeEnvironment:
    def __init__(self, maze, init_position, goal):
        # store dimensions
        rows = len(maze)
        cols = len(maze)
        self.boundary = np.array([rows, cols])

        # set start and goal
        self.init_position = init_position
        self.current_position = np.array(init_position)
        self.goal = goal
        self.maze = maze

        # keep track of visited positions
        self.visited = set()
        self.visited.add(tuple(self.current_position))

        # get all open (non-wall) cells
        self.allowed_states = np.array(np.where(maze == 0)).T.tolist()
        self.distances = np.linalg.norm(np.array(self.allowed_states) - np.array(goal), axis=1)

        # remove goal from allowed states and distances
        goal_idx = np.where(self.distances == 0)[0][0]
        del self.allowed_states[goal_idx]
        self.distances = np.delete(self.distances, goal_idx)

        # define action mappings: right, left, down, up
        self.action_map = {
            0: [0, 1],   # →
            1: [0, -1],  # ←
            2: [1, 0],   # ↓
            3: [-1, 0]   # ↑
        }

        # symbols for plotting arrows
        self.directions = {
            0: '→',
            1: '←',
            2: '↓',
            3: '↑'
        }

    # reset policy to bias start near goal when epsilon is high
    def reset_policy(self, eps, reg=7):
        return sp.softmax(-self.distances / (reg * (1 - eps**(2 / reg)))**(reg / 2)).squeeze()

    # reset environment after episode
    def reset(self, epsilon, prand=0):
        if np.random.rand() < prand:
            idx = np.random.choice(len(self.allowed_states))
        else:
            probs = self.reset_policy(epsilon)
            idx = np.random.choice(len(self.allowed_states), p=probs)

        self.current_position = np.array(self.allowed_states[idx])
        self.visited = {tuple(self.current_position)}
        return self.state()

    # take one step and return (next_state, reward, is_game_on)
    def state_update(self, action):
        is_game_on = True
        reward = -0.05  # step penalty

        move = self.action_map[action]
        next_pos = self.current_position + np.array(move)

        # if goal is reached
        if np.array_equal(self.current_position, self.goal):
            return [self.state(), 1, False]

        # if cell is visited again
        if tuple(self.current_position) in self.visited:
            reward = -0.2

        # if move is valid, go there; else, penalize
        if self.is_state_valid(next_pos):
            self.current_position = next_pos
        else:
            reward = -1

        self.visited.add(tuple(self.current_position))
        return [self.state(), reward, is_game_on]

    # return state with agent encoded as 2
    def state(self):
        state = copy.deepcopy(self.maze)
        state[tuple(self.current_position)] = 2
        return state

    # check if position is out of bounds
    def check_boundaries(self, position):
        pos = np.array(position)
        return np.any(pos < 0) or np.any(self.boundary - pos <= 0)

    # check if position is a wall
    def check_walls(self, position):
        return self.maze[tuple(position)] == 1

    # check if position is valid
    def is_state_valid(self, position):
        return not self.check_boundaries(position) and not self.check_walls(position)

    # draw maze with agent and goal
    def draw(self, filename):
        plt.figure()
        plt.imshow(self.maze, cmap='Greys', interpolation='none', aspect='equal')
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])

        # goal in blue, agent in red
        ax.plot(self.goal[1], self.goal[0], 'bs', markersize=4)
        ax.plot(self.current_position[1], self.current_position[0], 'rs', markersize=4)

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
