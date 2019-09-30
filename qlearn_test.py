# Q-Learning Test Algorithm
# Created as a sanity check for a Q-Learning framework in one of two semester projects for CS 5033 (Machine Learning)
# Author: Cameron Bost
import calendar
import time
import os
from os.path import exists

import numpy as np

import gym

# Flag for indicating when program should terminate
do_terminate_qlearn = False

# Gym instance
env = gym.make('CarRacing-v0')

# Environment properties
action_space_size = np.prod(env.action_space.shape)
observation_space_size = np.prod(env.observation_space.shape)

# Q-Learning Parameters
# TODO revise hyperparameters
ALPHA = 0.01
GAMMA = .9
MAX_ITER_PER_EPISODE = 1000

# Q-Table, a matrix of actions x states that tracks previous reward values for all visited states
q_table = None


# Exports the current contents of the Q-Table to disk and creates a backup of the previous contents.
def export_qtable():
    global q_table
    q_file = 'q.txt'
    if exists(q_file):
        backup_file = 'q_backup.txt'
        os.rename(q_file, backup_file)
    np.savetxt(q_file, q_table)


# Attempts to load Q-Table data from file. If it fails, Q-Table is initialized with zeros.
# TODO Consider not using zeros for the initial values
def init_q_table():
    global q_table, env
    q_file = 'q.txt'
    if exists(q_file):
        try:
            q_table = np.loadtxt(q_file, dtype=float)
        except IOError as e:
            print("Failed to read from file:", e)
            q_table = np.zeros([np.prod(env.observation_space.shape), np.prod(env.action_space.shape)])
    else:
        q_table = np.zeros([np.prod(env.observation_space.shape), np.prod(env.action_space.shape)])


# Performs one entire episode (aka epoch) of Q-Learning training
# This method was inspired by an example Q-Learning algorithm located at:
# https://towardsdatascience.com/reinforcement-learning-with-openai-d445c2c687d2
def do_qlearn_episode():
    global do_terminate_qlearn
    print("Initializing Q-Table...")
    init_q_table()
    # Initial state is determined by environment
    current_state = env.reset()
    iteration_ctr = 0
    # We track the individual reward at each stage to observe trends in the agent's immediate success
    current_reward_list = list()
    # We track the cumulative reward to observe the agent's overall success
    cumulative_reward_list = list()
    current_cumulative_reward = 0

    print("Beginning training episode...")
    # Repeat until max iterations have passed or agent reaches terminal state
    while not do_terminate_qlearn and iteration_ctr <= MAX_ITER_PER_EPISODE:
        # Keep that UI train rolling
        env.render()

        # Choose action. We add artificial stochastic variance to Q-Table to encourage deviance from the Q-Table
        exploration_motivator = np.random.randn(1, action_space_size)  # TODO investigate usefulness of: *(1./(i+1))
        selected_action = np.argmax(q_table[current_state, :] + exploration_motivator)

        # Perform action, update state
        next_state, reward, do_terminate_qlearn, info = env.step(selected_action)

        # Update Q-Table w/ standard Q-Learning rule
        d_q = ALPHA * (reward + GAMMA * np.max(q_table[next_state, :]) - q_table[current_state, selected_action])
        q_table[current_state, selected_action] += d_q

        # Update agent state variables
        current_state = next_state
        iteration_ctr += 1

        # Update reward trackers
        current_reward_list.append(reward)
        current_cumulative_reward += reward
        cumulative_reward_list.append(current_cumulative_reward)

        # After some number of iterations we automatically terminate the episode
        if iteration_ctr > MAX_ITER_PER_EPISODE:
            print("Note: Terminating episode due to max iterations exceeded")
            do_terminate_qlearn = True
    # Export results to file
    print("Exporting results...")
    fname = '%d_rewards' % calendar.timegm(time.gmtime())
    np.savetxt(fname+".txt", current_reward_list)
    np.savetxt(fname+"_cum.txt", cumulative_reward_list)
    # Export Q-Table contents to file
    print("Exporting Q-Table...")
    export_qtable()
    print("Episode completed successfully")


do_qlearn_episode()
