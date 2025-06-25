# Maze_solver_DQN
# 🧠 Deep Q-Learning Maze Solver

This project implements a Deep Q-Learning agent to solve a randomly generated maze. The agent uses a fully connected neural network (`fc_nn`) to approximate Q-values for navigation decisions.

---

## 🚀 Features

- 🔁 Deep Q-Learning with experience replay
- 🎲 Epsilon-decay for exploration vs. exploitation
- 🎮 Softmax and epsilon-greedy action selection
- 📊 Visualizations for:
  - Epsilon decay
  - Policy map
  - Loss trends
  - Reward heatmap
- 🧠 Agent learns from raw maze state
---

## ⚙️ How It Works

1. **Maze Generation**  
   A random 2D maze is generated using DFS backtracking, saved as a NumPy array.

2. **Environment**  
   - Agent interacts with the maze via four actions (up/down/left/right).
   - Invalid moves penalized, goal rewarded.

3. **Agent (DQN)**  
   - Stores experiences in a replay buffer.
   - Selects actions using either softmax or ε-greedy.
   - Learns Q-values using MSE loss.

4. **Training Loop**  
   - Trains for a defined number of epochs.
   - Periodically saves best-performing model.
   - Visualizations generated for policy, loss, and exploration.

---
