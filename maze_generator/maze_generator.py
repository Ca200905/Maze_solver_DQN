import random
import matplotlib.pyplot as plt
import numpy as np

# define maze size
width =10 
height = 10

# initialize maze with all zeros (walls)
maze = [[0 for _ in range(width)] for _ in range(height)]

# movement directions: right, down, left, up
dx = [0, 1, 0, -1]
dy = [-1, 0, 1, 0]

# start from a random cell
x = random.randint(0, width - 1)
y = random.randint(0, height - 1)
maze[y][x] = 1

# stack to store path: (x, y, direction)
stack = [(x, y, 0)]

# generate maze using randomized dfs
while stack:
    x, y, last_dir = stack[-1]

    # prevent zigzags by locking direction if previous move changed direction
    if len(stack) > 2:
        if last_dir != stack[-2][2]:
            direction_range = [last_dir]
        else:
            direction_range = range(4)
    else:
        direction_range = range(4)

    # find valid neighboring cells
    neighbors = []
    for direction in direction_range:
        nx = x + dx[direction]
        ny = y + dy[direction]

        # check if inside maze and unvisited
        if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] == 0:
            count = 0  # count adjacent visited cells
            for j in range(4):
                ex = nx + dx[j]
                ey = ny + dy[j]
                if 0 <= ex < width and 0 <= ey < height:
                    if maze[ey][ex] == 1:
                        count += 1
            # only allow if it connects to exactly one visited cell
            if count == 1:
                neighbors.append(direction)

    # if valid neighbors found, move to one of them
    if neighbors:
        chosen_dir = random.choice(neighbors)
        x += dx[chosen_dir]
        y += dy[chosen_dir]
        maze[y][x] = 1
        stack.append((x, y, chosen_dir))
    else:
        stack.pop()

# convert to numpy array and format walls as 1, path as 0
maze = np.array(maze)
maze = abs(maze - 1)

# ensure start and goal are path (not walls)
maze[0][0] = 0
maze[height - 1][width - 1] = 0

# save the maze
np.save('maze', maze)
