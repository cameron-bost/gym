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
from random import choices, getrandbits
import sys

EXPERIMENT = "double_test"
cur_path = os.path.dirname(__file__)
q1_file = f'{EXPERIMENT}_q1.gz'
q1_path = os.path.join(cur_path, q1_file)
q2_file = f'{EXPERIMENT}_q2.gz'
q2_path = os.path.join(cur_path, q2_file)

# Flag for indicating when program should terminate
do_terminate = False

# Gym instance
env = gym.make('CarRacing5033Discrete-v0')

# Environment properties
observation_space_size = np.prod(env.observation_space.nvec)
action_space_size = np.prod(env.action_space.nvec)

# Running Parameters
START_EPISODE = 0
Q_SAVE_INTERVAL = 100
Q_BACKUPS = False
STATS_SAVE_INTERVAL = 50
VIEW_INTERVAL = -1

# Q-Learning Parameters
# TODO revise hyperparameters
ALPHA_INIT = 0.7
ALPHA_MIN = 0.05
ALPHA_MIN_EPISODE = 5000
GAMMA_INIT = 0.5
GAMMA_MAX = 0.9
GAMMA_MAX_EPISODE = 2500
NUM_EPISODES = 10000
EPISODE_ITERATION_INIT = 500
EPISODE_ITERATION_MAX = 1000
EPISODE_ITERATION_MAX_EPISODE = 5000
EXPLORE_RATE_MAX = 0.5
EXPLORE_RATE_MIN = 0.1
EXPLORE_RATE_MIN_EPISODE = 5000
EXPLORE_DECAY_RATE = 1 - (EXPLORE_RATE_MIN/EXPLORE_RATE_MAX)**(1/EXPLORE_RATE_MIN_EPISODE)

#EXPERIMENTAL
ITER_COMPLETION_CHECKPOINT_PERCENT = 0.5
ITER_COMPLETION_CHECKPOINT_INCREMENT = 100

# Q-Table, an array of actions x states that tracks previous reward values for all visited states
q1_table = None
q2_table = None

# Exports the current contents of the Q-Table to disk and creates a backup of the previous contents.
def export_qtable(episode_num, episode_tiles_per_iter_list, tiles_visited_list, double):
    global q1_table, q2_table, cur_path, q1_path, q2_path

    if episode_num % Q_SAVE_INTERVAL == 0 and episode_num > 0:
        print("Exporting Q-Tables...")
        np.savetxt(q1_path, q1_table)
        if double: np.savetxt(q2_path, q2_table)
        if Q_BACKUPS:
            backup_file = f'q_backups\\{EXPERIMENT}_ep{episode_num}_q1.gz'
            backup_path = os.path.join(cur_path, backup_file)
            if exists(backup_path):
                os.remove(backup_path)
            copyfile(q1_path, backup_path)

            if double:
                backup_file = f'q_backups\\{EXPERIMENT}_ep{episode_num}_q2.gz'
                backup_path = os.path.join(cur_path, backup_file)
                if exists(backup_path):
                    os.remove(backup_path)
                copyfile(q2_path, backup_path)

    if episode_num % STATS_SAVE_INTERVAL == 0 and episode_num > 0:
        print("Exporting Rewards...")
        directory = os.path.join(cur_path, f"rewards\\{EXPERIMENT}\\")
        if not os.path.exists(directory):
            os.makedirs(directory)
        tiles_per_iter_dir = os.path.join(directory, f"tiles_per_iter\\")
        if not os.path.exists(tiles_per_iter_dir):
            os.makedirs(tiles_per_iter_dir)
        fname = f'{EXPERIMENT}_ep{episode_num-STATS_SAVE_INTERVAL+1}_{episode_num}_tiles_per_iter_per_ep'
        np.savetxt(tiles_per_iter_dir+fname+".gz", episode_tiles_per_iter_list)
        if tiles_visited_list:
            tiles_visited_dir = os.path.join(directory, f"tiles_visited\\")
            if not os.path.exists(tiles_visited_dir):
                os.makedirs(tiles_visited_dir)
            fname = f'{EXPERIMENT}_ep{episode_num}_tiles'
            np.savetxt(tiles_visited_dir+fname+".gz", tiles_visited_list)

        return True
    else:
        return False


# Attempts to load Q-Table data from file. If it fails, Q-Table is initialized with zeros.
# TODO Consider not using zeros for the initial values
def init_q_table(q_path):
    global env, cur_path
    q_table = None
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
    return q_table

def init_q_tables(double):
    global q1_table, q2_table, q1_path, q2_path
    q1_table = init_q_table(q1_path)
    if double: q2_table = init_q_table(q2_path)

# Performs one entire episode (aka epoch) of Q-Learning training
# This method was inspired by an example Q-Learning algorithm located at:
# https://towardsdatascience.com/reinforcement-learning-with-openai-d445c2c687d2


def do_qlearn_episode(episode_num, policy, learning_method, max_iterations, gamma, q_table, alpha):
    global do_terminate
    # Initial state is determined by environment
    current_state = env.reset()
    iteration_ctr = 0

    tiles_visited_list = []
    tiles_per_iter = 0
    current_average_reward = 0

    # Repeat until max iterations have passed or agent reaches terminal state
    while not do_terminate_qlearn and iteration_ctr <= max_iterations:

        # Get current state index
        current_state_index = np.ravel_multi_index(current_state, env.observation_space.nvec)
        selected_action_index = policy(current_state_index, q_table)
        selected_action = np.unravel_index(
            selected_action_index, env.action_space.nvec)


        # Perform action, update state
        next_state, reward, do_terminate_qlearn, tile_visited_count = env.step(selected_action)

        # Set gamma = 0 when terminal state reached to satisfy Q(s', . ) = 0
        if do_terminate_qlearn:
            gamma = 0

        # Update Q-Table w/ selected learning method
        next_state_index = np.ravel_multi_index(next_state, env.observation_space.nvec)
        learning_method(reward, current_state_index, selected_action_index,
                        next_state_index, gamma, q_table, alpha)

        # Update agent state variables
        current_state = next_state
        iteration_ctr += 1

        # Update reward trackers, only use tile visited so that the reward function can be evaluated independently
        tiles_visited_list.append(tile_visited_count)
        tiles_per_iter = tile_visited_count / iteration_ctr
        current_average_reward = (current_average_reward*(iteration_ctr-1) + reward)/iteration_ctr

        # After some number of iterations we automatically terminate the episode
        if iteration_ctr > max_iterations:
            # print("Note: Terminating episode due to max iterations exceeded")
            do_terminate_qlearn = True
    print(
        f"Episode {episode_num} completed. Tiles per iter: {tiles_per_iter:.4f}, Avg Reward: {current_average_reward:.4f}, Iter: {iteration_ctr}, Tiles:{tiles_visited_list[-1]}")
    return tiles_visited_list, tiles_per_iter

def expected_sarsa(reward, current_state_index, selected_action_index, next_state_index, gamma, q_table, alpha):
    expected_val = expected_action_value(q_table, next_state_index, greedy_policy, exponential_weights)
    d_q = alpha * (reward
                   + gamma * expected_val
                   - q_table[current_state_index, selected_action_index])
    q_table[current_state_index, selected_action_index] += d_q


def double_expected_sarsa(reward, current_state_index, selected_action_index, next_state_index, gamma, q_tables, alpha):
    q1_table, q2_table = q_tables
    d_q = None
    # Randomly choose to update one of the q_tables
    use_q1_table = bool(getrandbits(1))
    if use_q1_table:
        expected_val = expected_action_value(q2_table, next_state_index, greedy_policy, exponential_weights)
        d_q = alpha * (reward
                    + gamma * expected_val
                    - q1_table[current_state_index, selected_action_index])
        q1_table[current_state_index, selected_action_index] += d_q
    else:
        expected_val = expected_action_value(q1_table, next_state_index, greedy_policy, exponential_weights)
        d_q = alpha * (reward
                       + gamma * expected_val
                       - q2_table[current_state_index, selected_action_index])
        q2_table[current_state_index, selected_action_index] += d_q

def expected_action_value(q_table, next_state_index, policy, weight_fn):
    # Given Q table, next state, policy, and weight function, returns the expected value for the next state
    action_list = q_table[next_state_index]
    weights = weight_fn(next_state_index, policy, q_table)
    expected_val = [weight*action for weight,
                  action in zip(weights, action_list)]
    return sum(expected_val)

def greedy_policy(current_state_index, q_table):
    # Choose action with highest reward from given q table
    best_action_index = np.argmax(q_table[current_state_index])
    return best_action_index

def double_greedy_policy(current_state_index, q_tables):
    # Choose action with highest reward from both q tables
    q1_table, q2_table = q_tables
    qsum = q1_table[current_state_index]+q2_table[current_state_index]
    best_action_index = np.argmax(qsum)
    return best_action_index


def exponential_explore_policy(current_state_index, q_table):
    # Get action given exponential weights
    weights = exponential_weights(current_state_index, greedy_policy, q_table)
    # Generate list from [0, ..., action_space_size)
    pop = [i for i in range(action_space_size)]
    selected_action_index = choices(pop, weights)[0]
    return selected_action_index

def double_exponential_explore_policy(current_state_index, q_table):
    # Get action given exponential weights and given q_table
    weights = exponential_weights(current_state_index, double_greedy_policy, q_table)
    # Generate list from [0, ..., action_space_size)
    pop = [i for i in range(action_space_size)]
    selected_action_index = choices(pop, weights)[0]
    return selected_action_index

def exponential_weights(current_state_index, policy, q_table):
    global exploration_rate
    # Choose action with highest reward
    best_action_index = policy(current_state_index, q_table)
    # Generate list from [0, ..., action_space_size)
    pop = [i for i in range(action_space_size)]
    # Get weights for non-best-valued action
    individual_weight = exploration_rate/(action_space_size-1)
    # Generate list of weights
    weights = [individual_weight if a != best_action_index
               else (1-exploration_rate) for a in pop]
    return weights


def do_policy_episode(policy, max_iterations, q_table, render):
    global do_terminate
    # Perform a run using the given policy, but don't update q table.
    # Initial state is determined by environment
    current_state = env.reset()
    iteration_ctr = 0

    tiles_visited_list = []
    tiles_per_iter = 0
    current_average_reward = 0

    # Repeat until max iterations have passed or agent reaches terminal state
    while not do_terminate_qlearn and iteration_ctr <= max_iterations:
        # Keep that UI train rolling
        if render:
            env.render()

        # Get current state index
        current_state_index = np.ravel_multi_index(current_state, env.observation_space.nvec)
        selected_action_index = policy(current_state_index, q_table)
        selected_action = np.unravel_index(selected_action_index, env.action_space.nvec)

        # Perform action, update state
        next_state, reward, do_terminate_qlearn, tile_visited_count = env.step(selected_action)

        # Update agent state variables
        current_state = next_state
        iteration_ctr += 1

        # Update reward trackers, only use tile visited so that the reward function can be evaluated independently
        tiles_visited_list.append(tile_visited_count)
        tiles_per_iter = tile_visited_count / iteration_ctr
        current_average_reward = (
            current_average_reward*(iteration_ctr-1) + reward)/iteration_ctr


        # After some number of iterations we automatically terminate the episode
        if iteration_ctr > max_iterations:
            # print("Note: Terminating episode due to max iterations exceeded")
            do_terminate_qlearn = True
    return tiles_visited_list, tiles_per_iter


def do_human(max_iterations, actions):
    global do_terminate
    # Perform a run using the human
    current_state = env.reset()
    iteration_ctr = 0

    tiles_visited_list = []
    tiles_per_iter = 0
    current_average_reward = 0

    # Repeat until max iterations have passed or agent reaches terminal state
    while not do_terminate_qlearn and iteration_ctr <= max_iterations:
        # Keep that UI train rolling
        env.render()

        # Perform action, update state
        next_state, reward, do_terminate_qlearn, tile_visited_count = env.step(actions)

        # Update agent state variables
        current_state = next_state
        iteration_ctr += 1

        # Update reward trackers, only use tile visited so that the reward function can be evaluated independently
        tiles_visited_list.append(tile_visited_count)
        tiles_per_iter = tile_visited_count / iteration_ctr
        current_average_reward = (
            current_average_reward*(iteration_ctr-1) + reward)/iteration_ctr

        # After some number of iterations we automatically terminate the episode
        if iteration_ctr > max_iterations:
            # print("Note: Terminating episode due to max iterations exceeded")
            do_terminate_qlearn = True
    return tiles_visited_list, tiles_per_iter

if __name__ == "__main__":

    if len(sys.argv) > 1:
        try:
            if sys.argv[1] == "human":
                # Keyboard stuff
                from pyglet.window import key
                actions = np.array([0.0, 0.0, 0.0])

                def key_press(k, mod):
                    if k == key.LEFT:
                        actions[0] = 1
                    if k == key.RIGHT:
                        actions[0] = 2
                    if k == key.UP:
                        actions[1] = 1
                    if k == key.DOWN:
                        actions[2] = 1
                    if k == key.ESCAPE or k == key.SPACE:
                        sys.exit(0)

                def key_release(k, mod):
                    if k == key.LEFT and actions[0] == 1:
                        actions[0] = 0
                    if k == key.RIGHT and actions[0] == 2:
                        actions[0] = 0
                    if k == key.UP:
                        actions[1] = 0
                    if k == key.DOWN:
                        actions[2] = 0
                env.render()
                env.viewer.window.on_key_press = key_press
                env.viewer.window.on_key_release = key_release
                while True:
                    tiles_visited_list, tiles_per_iter = do_human(EPISODE_ITERATION_MAX, actions)
                    do_terminate = False
                    print(f"Human episode complete. Iter:{len(tiles_visited_list)}, Tiles per iter: {tiles_per_iter:.4f}, Tiles:{tiles_visited_list[-1]}")
            else:
                double = "double" in sys.argv
                str_add = ""
                if double:
                    str_add = "Double "

                print(f"Initializing {str_add}Q-Table...")
                init_q_tables(double)

                #Increment number of iterations per episode to try to learn to go straight first then around corners
                iter_incr = (EPISODE_ITERATION_MAX - EPISODE_ITERATION_INIT)/EPISODE_ITERATION_MAX_EPISODE
                iterations = min(EPISODE_ITERATION_MAX, EPISODE_ITERATION_INIT + iter_incr * START_EPISODE)

                # From wiki, start low then end high
                gamma_incr = (GAMMA_MAX-GAMMA_INIT)/GAMMA_MAX_EPISODE
                gamma = min(GAMMA_MAX, GAMMA_INIT + gamma_incr * START_EPISODE)

                # Learn more at first, then learn more minute details later
                alpha_incr = (ALPHA_MIN - ALPHA_INIT)/ALPHA_MIN_EPISODE
                alpha = max(ALPHA_MIN, ALPHA_INIT + alpha_incr * START_EPISODE)

                episode_tiles_per_iter_list = []
                tiles_visited_list = None

                if "greedy" in sys.argv:
                    while True:
                        if double:
                            tiles_visited_list, tiles_per_iter = do_policy_episode(double_greedy_policy, EPISODE_ITERATION_MAX, (q1_table, q2_table), render=True)
                        else:
                            tiles_visited_list, tiles_per_iter = do_policy_episode(greedy_policy, EPISODE_ITERATION_MAX, q1_table, render=True)
                        do_terminate = False
                        print(f"{str_add}Greedy episode complete. Iter:{len(tiles_visited_list)}, Tiles per iter: {tiles_per_iter:.4f}, Tiles:{tiles_visited_list[-1]}")
                elif "learn" in sys.argv:
                    for episode in range(START_EPISODE, NUM_EPISODES):
                        # Perform increments
                        # Exploration rate; exponential decay
                        exploration_rate = max(EXPLORE_RATE_MAX*(1-EXPLORE_DECAY_RATE)**episode, EXPLORE_RATE_MIN)
                        iterations = min(EPISODE_ITERATION_MAX, iterations + iter_incr)
                        gamma = min(GAMMA_MAX, gamma + gamma_incr)
                        alpha = max(ALPHA_MIN, alpha + alpha_incr)
                        print(f"Performing run #{episode}. Iterations:{iterations:.0f}, Alpha:{alpha:.4f}, Gamma:{gamma:.4f}, Epsilon:{exploration_rate:.4f}")
                        if double:
                            tiles_visited_list, tiles_per_iter = do_qlearn_episode(
                                episode, double_exponential_explore_policy, double_expected_sarsa, iterations, gamma, (q1_table, q2_table), alpha)
                        else:
                            tiles_visited_list, tiles_per_iter = do_qlearn_episode(
                                episode, exponential_explore_policy, expected_sarsa, iterations, gamma, q1_table, alpha)
                        iters_complete = len(tiles_visited_list)
                        tiles_visited_list = None # Don't save stats for non-greedy policy runs
                        do_terminate = False
                        episode_tiles_per_iter_list.append(tiles_per_iter)

                        show_user = episode > 0 and VIEW_INTERVAL > 0 and episode % VIEW_INTERVAL == 0
                        save_stats = episode > 0 and episode % STATS_SAVE_INTERVAL == 0
                        if save_stats or show_user:
                            # show the viewer what we've learned so far
                            if double:
                                tiles_visited_list, tiles_per_iter = do_policy_episode(
                                    double_greedy_policy, iterations, (q1_table, q2_table), render=show_user)
                            else:
                                tiles_visited_list, tiles_per_iter = do_policy_episode(
                                    greedy_policy, iterations, q1_table, render=show_user)
                            if not save_stats:
                                tiles_visited_list = None
                            # print(f"{str_add}Greedy episode {episode} complete. Iter:{len(tiles_visited_list)}, Tiles per iter: {tiles_per_iter:.4f}, Tiles:{tiles_visited_list[-1]}")
                            do_terminate = False

                        if export_qtable(episode, episode_tiles_per_iter_list, tiles_visited_list, double):
                            episode_tiles_per_iter_list = []
                            tiles_visited_list = None

                        print()
        except KeyboardInterrupt:
            sys.exit(0)





