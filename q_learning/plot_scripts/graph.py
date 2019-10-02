# Plots reward results over time
import matplotlib.pyplot as plt, numpy as np, os
results_file = 'averages_1_2100.txt'
reward_values = np.loadtxt(results_file, dtype=float)
fig = plt.figure(0)
fig.canvas.set_window_title('CS 4033/5033 - Cameron Bost, Bryce Hennen')
plt.plot(range(len(reward_values)),reward_values)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('E-SARSA (Cumulative), alpha=0.15')
plt.show()