# Q-Learning Test Algorithm
# Created as a sanity check for a Q-Learning framework in one of two semester projects for CS 5033 (Machine Learning)
# Author: Cameron Bost
import calendar
import time
import os
from os.path import exists
from shutil import copyfile
import numpy as np
import gym
from random import choices

EXPERIMENT = "sensors"
cur_path = os.path.dirname(__file__)
q_file = f'{EXPERIMENT}_q.txt'
q_path = os.path.join(cur_path, q_file)

# Flag for indicating when program should terminate
do_terminate_qlearn = False

# Gym instance
env = gym.make('CarRacing-v0')

# Environment properties
observation_space_size = np.prod(env.observation_space.nvec)
action_space_size = np.prod(env.action_space.nvec)

# Q-Learning Parameters
# TODO revise hyperparameters
ALPHA = 0.01
GAMMA = .9
NUM_EPISODES = 5000
MAX_ITER_PER_EPISODE = 1000
MAX_EXPLORE_RATE=0.5
EXPLORE_DECAY_RATE=0.01
MIN_EXPLORE_RATE=0.01


# Q-Table, a matrix of actions x states that tracks previous reward values for all visited states
q_table = None


# Exports the current contents of the Q-Table to disk and creates a backup of the previous contents.
def export_qtable(episode_num):
    global q_table, cur_path, q_path
    backup_file = f'q_backups\\{EXPERIMENT}_ep{episode_num}_q.txt'
    backup_path = os.path.join(cur_path, backup_file)
    if exists(backup_path): 
        os.remove(backup_path)
    np.savetxt(q_path, q_table)
    copyfile(q_path, backup_path)


# Attempts to load Q-Table data from file. If it fails, Q-Table is initialized with zeros.
# TODO Consider not using zeros for the initial values
def init_q_table():
    global q_table, env, cur_path, q_path
    if exists(q_path):
        try:
            q_table = np.loadtxt(q_path, dtype=float)
            print("Using existing Q-Table...")
        except IOError as e:
            print("Failed to read from file:", e)
            q_table = np.zeros([observation_space_size, action_space_size])
    else:
        print("Creating new Q-Table...")
        q_table=np.zeros([observation_space_size, action_space_size])


# Performs one entire episode (aka epoch) of Q-Learning training
# This method was inspired by an example Q-Learning algorithm located at:
# https://towardsdatascience.com/reinforcement-learning-with-openai-d445c2c687d2
def do_qlearn_episode(episode_num):
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

#Get random exploration chance (chance to deviate from reward in different action)
    exploration_rate = max(MAX_EXPLORE_RATE*(1-EXPLORE_DECAY_RATE)**episode_num, MIN_EXPLORE_RATE) #Exponential decay of exploration

    print(f"Beginning training episode {episode_num}...")
    # Repeat until max iterations have passed or agent reaches terminal state
    while not do_terminate_qlearn and iteration_ctr <= MAX_ITER_PER_EPISODE:
        # Keep that UI train rolling
        env.render()

        # Choose action. We add artificial stochastic variance to Q-Table to encourage deviance from the Q-Table
        # exploration_motivator = np.random.randn(1, action_space_size)*(1./(episode_num+1))  # TODO investigate usefulness of: *(1./(i+1))
        
        # Choose action with highest reward
        current_state_index = np.ravel_multi_index(current_state, env.observation_space.nvec)
        best_action_index = np.argmax(q_table[current_state_index])
        
        # Generate list from [0, ..., action_space_size)
        pop = [i for i in range(action_space_size)]
        # Get weights for non-best-valued action
        individual_weight = exploration_rate/(action_space_size-1)
        # Generate list of weights
        weights = [individual_weight if a != best_action_index
                   else (1-exploration_rate) for a in pop]
        selected_action_index = choices(pop, weights)[0]
        
        selected_action = np.unravel_index(selected_action_index, env.action_space.nvec)
        
        # Perform action, update state
        next_state, reward, do_terminate_qlearn, info = env.step(selected_action)

        # Update Q-Table w/ standard Q-Learning rule
        next_state_index = np.ravel_multi_index(next_state, env.observation_space.nvec)
        d_q = ALPHA * (reward + GAMMA * np.max(q_table[next_state_index]) - q_table[current_state_index, selected_action_index])
        q_table[current_state_index, selected_action_index] += d_q

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
    path = os.path.join(cur_path, "rewards\\")
    fname = f'{EXPERIMENT}_ep{episode_num}_rewards'
    np.savetxt(path+fname+".txt", current_reward_list)
    np.savetxt(path+fname+"_cum.txt", cumulative_reward_list)
    # Export Q-Table contents to file
    print("Exporting Q-Table...")
    export_qtable(episode_num)
    print("Episode completed successfully")


if __name__ == "__main__":
    start_episode=0
    do_qlearn_episode(start_episode)
    
    # for episode in range(start_episode, NUM_EPISODES):
    #     do_qlearn_episode(episode)
    #     do_terminate_qlearn = False
    

