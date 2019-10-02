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
q_file = f'{EXPERIMENT}_q.gz'
q_path = os.path.join(cur_path, q_file)

# Flag for indicating when program should terminate
do_terminate_qlearn = False

# Gym instance
env = gym.make('CarRacing-v0')

# Environment properties
observation_space_size = np.prod(env.observation_space.nvec)
action_space_size = np.prod(env.action_space.nvec)

# Running Parameters
start_episode = 0
SAVE_INTERVAL = 100
VIEW_INTERVAL = 100

# Q-Learning Parameters
# TODO revise hyperparameters
ALPHA = 0.15
GAMMA_INIT = 0.4
GAMMA_MAX = 0.85
GAMMA_MAX_EPISODE = 5000
NUM_EPISODES = 50000
EPISODE_ITERATION_INIT = 100
EPISODE_ITERATION_INCREMENT = 100
EPISODE_ITERATION_INCREMENT_INTERVAL = 250
MAX_ITER_PER_EPISODE = 2000
MAX_EXPLORE_RATE=0.5
EXPLORE_DECAY_RATE=0.001
MIN_EXPLORE_RATE=0.01


# Q-Table, an array of actions x states that tracks previous reward values for all visited states
q_table = None

# Exports the current contents of the Q-Table to disk and creates a backup of the previous contents.
def export_qtable(episode_num, episode_reward_list, cumulative_reward_list):
    global q_table, cur_path, q_path

    if episode_num % SAVE_INTERVAL == 0 and episode_num > 0:
        print("Exporting results...")
        print("Exporting Q-Table...")
        backup_file = f'q_backups\\{EXPERIMENT}_ep{episode_num}_q.gz'
        backup_path = os.path.join(cur_path, backup_file)
        if exists(backup_path): 
            os.remove(backup_path)
        np.savetxt(q_path, q_table)
        copyfile(q_path, backup_path)

        print("Exporting Rewards...")
        path = os.path.join(cur_path, "rewards\\")
        fname = f'{EXPERIMENT}_ep{episode_num-SAVE_INTERVAL+1}_{episode_num}_rewards'
        np.savetxt(path+fname+".gz", episode_reward_list)
        fname = f'{EXPERIMENT}_ep{episode_num}_rewards'
        np.savetxt(path+fname+".gz", cumulative_reward_list)

        return True
    else:
        return False


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
def do_qlearn_episode(episode_num, policy, learning_method, max_iterations, gamma):
    global do_terminate_qlearn
    # Initial state is determined by environment
    current_state = env.reset()
    iteration_ctr = 0
    # We track the individual reward at each stage to observe trends in the agent's immediate success
    current_reward_list = list()
    # We track the cumulative reward to observe the agent's overall success
    cumulative_reward_list = list()
    current_average_reward = 0

    # Repeat until max iterations have passed or agent reaches terminal state
    while not do_terminate_qlearn and iteration_ctr <= max_iterations:
        # Keep that UI train rolling
        if VIEW_INTERVAL > 0 and episode_num % VIEW_INTERVAL == 0:
            env.render()

        # Get current state index
        current_state_index = np.ravel_multi_index(current_state, env.observation_space.nvec)
        selected_action_index = policy(current_state_index)
        selected_action = np.unravel_index(
            selected_action_index, env.action_space.nvec)
        
        
        # Perform action, update state
        next_state, reward, do_terminate_qlearn, info = env.step(selected_action)

        # Update Q-Table w/ standard Q-Learning rule
        next_state_index = np.ravel_multi_index(next_state, env.observation_space.nvec)
        d_q = learning_method(reward, current_state_index, selected_action_index, next_state_index, gamma)
        q_table[current_state_index, selected_action_index] += d_q

        # Update agent state variables
        current_state = next_state
        iteration_ctr += 1

        # Update reward trackers
        current_reward_list.append(reward)
        current_average_reward = (current_average_reward*(iteration_ctr-1) + reward)/iteration_ctr
        cumulative_reward_list.append(current_average_reward)

        # After some number of iterations we automatically terminate the episode
        if iteration_ctr > max_iterations:
            # print("Note: Terminating episode due to max iterations exceeded")
            do_terminate_qlearn = True
    print(f"Episode {episode_num} completed. Average score: {current_average_reward}")
    return current_average_reward, cumulative_reward_list

def expected_sarsa(reward, current_state_index, selected_action_index, next_state_index, gamma):
    action_list = q_table[next_state_index]
    weights = exponential_weights(next_state_index)
    expected_q = [weight*action for weight, action in zip(weights,action_list)]
    epected_val = sum(expected_q)
    d_q = ALPHA * (reward 
                   + gamma * epected_val
                   - q_table[current_state_index, selected_action_index])
    return d_q

def greedy(current_state_index):
    # Choose action with highest reward
    best_action_index = np.argmax(q_table[current_state_index])
    return best_action_index

def exponential_explore(current_state_index):
    # Get action given exponential weights
    weights = exponential_weights(current_state_index)
    # Generate list from [0, ..., action_space_size)
    pop = [i for i in range(action_space_size)]
    selected_action_index = choices(pop, weights)[0]
    return selected_action_index

def exponential_weights(current_state_index):
    global exploration_rate
    # Choose action with highest reward
    best_action_index = greedy(current_state_index)
    # Generate list from [0, ..., action_space_size)
    pop = [i for i in range(action_space_size)]
    # Get weights for non-best-valued action
    individual_weight = exploration_rate/(action_space_size-1)
    # Generate list of weights
    weights = [individual_weight if a != best_action_index
               else (1-exploration_rate) for a in pop]
    return weights

if __name__ == "__main__":
    print("Initializing Q-Table...")
    init_q_table()
    # do_qlearn_episode(start_episode)

    max_iterations = EPISODE_ITERATION_INIT + (start_episode // EPISODE_ITERATION_INCREMENT_INTERVAL) * EPISODE_ITERATION_INCREMENT
    gamma_incr = (GAMMA_MAX-GAMMA_INIT)/GAMMA_MAX_EPISODE # From wiki, start low then end high
    gamma = min(GAMMA_MAX, GAMMA_INIT + gamma_incr * start_episode)
    episode_reward_list = []
    cumulative_reward_list = None
    for episode in range(start_episode, NUM_EPISODES):
        if episode % EPISODE_ITERATION_INCREMENT_INTERVAL == 0:
            max_iterations += EPISODE_ITERATION_INCREMENT
        max_iterations = min(max_iterations, MAX_ITER_PER_EPISODE) # Cap num iterations
        # Exploration rate; exponential decay
        exploration_rate = min(MAX_EXPLORE_RATE*(1-EXPLORE_DECAY_RATE)**episode, MIN_EXPLORE_RATE)
        gamma = min(GAMMA_MAX, gamma + gamma_incr)

        current_average_reward = do_qlearn_episode(
            episode, exponential_explore, expected_sarsa, max_iterations, gamma)
        do_terminate_qlearn = False
        episode_reward_list.append(current_average_reward)
        if export_qtable(episode, episode_reward_list, cumulative_reward_list):
            episode_reward_list = []
    
        do_terminate_qlearn = False
    

