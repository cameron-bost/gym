# Actor-Critic implementation
# Algorithm referenced from RL textbook p. 332 (PDF p. 354)
# This is being ran on a modified state representation (see car_racing_pos.py)
import math
import os
import sys
import numpy as np
import gym
import random 

"""
Output
"""
SHOW_GRAPHICS = True
SHOW_GRAPHICS_INTERVAL = 10 #Show episode every N episodes
SHOW_ACTIONS = False
SHOW_ACTIONS_INTERVAL = 1 #Show selected action every N iterations

"""
Learning Parameters
"""
MAX_EPISODES = 100000
EPISODE_MAX_ITERATIONS = 1000    # Max number of iterations in one episode
DEFAULT_GAMMA = 0.9
EXPLORATION_RATE = 0.1
VALUE_LEARNING_RATE = 0.005
POLICY_LEARNING_RATE = 0.005

"""
Model Constants
"""
ACTIONS = None
ACTION_COMBO_COEFFICIENTS = None    # Note: get [a_i][e_i], a_i in range(len(ACTIONS)), e_i in range(2**len(ACTIONS[0]))
COMBO_DIMENSION = None
ACTION_COMBOS = None

"""
Results Constants
"""
cur_path = os.path.dirname(__file__)
FILE_NAME_VALUE_WEIGHTS = os.path.join(cur_path, "value_weights.txt")
FILE_NAME_POLICY_WEIGHTS = os.path.join(cur_path, "policy_weights.txt")
DIRECTORY_WEIGHT_BACKUP = os.path.join(cur_path, "weight_backups")

# Gym instance
env = gym.make('CarRacing5033ContinuousState-v0')


# State constants (copied from gym)
RAY_CAST_DISTANCE = 20
NUM_SENSORS = 5
RAY_CAST_INTERVALS = 5

DISTANCE_INTERVALS = 6
SPEED_INTERVALS = 10  # Number of intervals to discretize speed state into
MAX_SPEED = 100.
MIN_SPEED = 0.

STEER_INTERVALS = 3
STEER_MAX = 0.4
STEER_MIN = -0.4


# Configures key listeners for gym window (allows human control of car)
def setup_controls():
    # Keyboard stuff
    from pyglet.window import key
    global actions

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


def do_human(max_iterations, actions_human):
    global do_terminate
    # Perform a run using the human
    env.reset()
    iteration_ctr = 0

    tiles_visited_list_human = []
    tiles_per_iter_human = 0
    current_average_reward = 0

    # Repeat until max iterations have passed or agent reaches terminal state
    while not do_terminate and iteration_ctr <= max_iterations:
        # Keep that UI train rolling
        env.render()

        # Perform action, update state
        next_state, reward, do_terminate, tile_visited_count = env.step(actions_human)
        iteration_ctr += 1

        # Update reward trackers, only use tile visited so that the reward function can be evaluated independently
        tiles_visited_list_human.append(tile_visited_count)
        tiles_per_iter_human = tile_visited_count / iteration_ctr
        current_average_reward = (
            current_average_reward*(iteration_ctr-1) + reward)/iteration_ctr

        # After some number of iterations we automatically terminate the episode
        if iteration_ctr > max_iterations:
            # print("Note: Terminating episode due to max iterations exceeded")
            do_terminate = True
    return tiles_visited_list_human, tiles_per_iter_human

# Evaluates a given state action pair given the active state features, the value for that state, 
# and the active action features.
def eval_state_action(state_exps, state, action_exps, action, state_val):
    # STATE =   {s0,s1,RAYS}, s0 = speed, s1 = steer_angle
    # ACTIONS = {steer, gas}
    # Raycasts ordered right to left in an arc
    enable_speed = state_exps[0]
    enable_steer_angle = state_exps[1]
    enable_raycasts = sum(state_exps[2:]) #enabled if at least one raycast
    enable_steer = action_exps[0]
    enable_gas = action_exps[1]

    steer, gas = action
    speed = state[0]
    # steer = [0,1], steer_angle =[-0.5,0.5]; right < 0, left > 0
    steer_angle = state[1] - 0.5
    #set raycast state value = 0 if that particular raycast is disabled
    raycasts = []
    for index, raycast in enumerate(state[2:]):
        if state_exps[index+2]:
            raycasts.append(raycast)
        else:
            raycasts.append(0)

    evaluation = state_val #TODO Does this mess with gradient?
    evaluation += -0.1
    # # positive speed reward
    if enable_speed: 
        evaluation += speed
        if speed < 0.1:
            if enable_gas:
                if gas == "gas":
                    evaluation += 1
                elif gas == "brake":
                    evaluation += -0.5

    # Maximize raycast distances
    if enable_raycasts:
        evaluation += sum(raycasts)/enable_raycasts

    # TODO decrease raycast score if higher speed (or middle raycast blocked)

    # TODO Brake gets increasingly better if closer to wall

    # TODO Base score on actual steering angle
    # Steer away from the closer raycasts
    # Goal is to maximize sum of raycasts (ie to have the most amount of raycasts pointing away from all)
    if enable_raycasts and enable_steer: #(enable_steer_angle or enable_steer):
        #Get left and right distances to wall
        split_pont = len(raycasts)//2
        right_raycasts = raycasts[:split_pont]
        left_raycasts = raycasts[-split_pont:]
        difference = sum(right_raycasts) - sum(left_raycasts)

        #Difference > 0 means right side further away; turn right to try to maximize
        #Difference < 0 means left side further away; turn left to try to maximize
        #Determine if current action and angle are correct
        #If angle is correct, set it positive, otherwise use negative
        #right: steer_angle < 0, left: steer_angle > 0
        steer_action_val = 0
        # steer_angle_val = 0

        if difference > 0:
            if steer == "right":
                steer_action_val = 0.1
            else:
                steer_action_val = -0.1
            # if steer_angle < 0:
            #     steer_angle_val = abs(steer_angle)
            # else:
            #     steer_angle_val = -abs(steer_angle)
        elif difference < 0:
            if steer == "left":
                steer_action_val = 0.1
            else:
                steer_action_val = -0.1
            # if steer_angle > 0:
            #     steer_angle_val = abs(steer_angle)
            # else:
            #     steer_angle_val = -abs(steer_angle)
        else:
            # No difference, so only straight is correct
            if steer == "straight":
                steer_action_val = 0.1
            else:
                steer_action_val = -0.1
            # steer_angle_val = 2
        # print(steer_action_val, steer, raycasts)

        # Score proportional to magnitude of difference
        raycast_val = abs(difference)
        
        #enable or disable calculated values
        # steer_angle_val **= enable_steer_angle

        evaluation = raycast_val*steer_action_val
        # evaluation += raycast_val*(steer_action_val+steer_angle_val)

    return evaluation

# A feature X that receives state as input when evaluating.
# Upon evaluation, this will return the state feature value (used as x(s)) as well as all possible policy feature
# values.
class StateFeature:
    def __init__(self, exps):
        self.exps = exps
        self.state_value = 0
        self.action_values = {}

    def eval(self, state):
        self.state_value = float(
            np.prod([state[i]**self.exps[i] for i in range(len(state))]))
        self.action_values = {}
        for action_idx, action in enumerate(ACTIONS):
            action_combo_values = []
            for action_exps in ACTION_COMBOS:
                action_combo_value = eval_state_action(
                    self.exps, state, action_exps, action, self.state_value)
                action_combo_values.append(action_combo_value)
            self.action_values[action] = action_combo_values
        return self.state_value, self.action_values

    def eval_action(self, a):
        return self.action_values[a]


# Hand-picked State Feature Vectors
# STATE =   {s0,s1,RAYS}, s0 = speed, s1 = steer_angle
# RAYS =    {r0,r1,..,r4}
# X(S) =    {1,
#            s0,
#            s1,
#            r0, .., r4,
#            s0*r0, s0*r1, .., s0*r4,
#            s1*r0, s1*r1, .., s1*r4}
STATE_FEATURES = [StateFeature(exps=[0, 0, 0, 0, 0, 0, 0]), # 1
                  StateFeature(exps=[1, 0, 0, 0, 0, 0, 0]), # s0
                  StateFeature(exps=[0, 1, 0, 0, 0, 0, 0]), # s1
                  StateFeature(exps=[1, 1, 0, 0, 0, 0, 0]), # s0*s1
                  StateFeature(exps=[1, 1, 1, 1, 1, 1, 1]), # s0*s1*r0*r1*r2*r3*r4
                  # RAYS
                  StateFeature(exps=[0, 0, 1, 1, 1, 1, 1]), # r0*r1*r2*r3*r4
                  StateFeature(exps=[0, 0, 1, 0, 0, 0, 0]), # r0 
                  StateFeature(exps=[0, 0, 0, 1, 0, 0, 0]), # r1
                  StateFeature(exps=[0, 0, 0, 0, 1, 0, 0]), # r2 
                  StateFeature(exps=[0, 0, 0, 0, 0, 1, 0]), # r3
                  StateFeature(exps=[0, 0, 0, 0, 0, 0, 1]), # r4
                  # s0 * RAYS
                  StateFeature(exps=[1, 0, 1, 1, 1, 1, 1]), # s0*r0*r1*r2*r3*r4
                  StateFeature(exps=[1, 0, 1, 0, 0, 0, 0]), # s0*r0 
                  StateFeature(exps=[1, 0, 0, 1, 0, 0, 0]), # s0*r1
                  StateFeature(exps=[1, 0, 0, 0, 1, 0, 0]), # s0*r2 
                  StateFeature(exps=[1, 0, 0, 0, 0, 1, 0]), # s0*r3
                  StateFeature(exps=[1, 0, 0, 0, 0, 0, 1]), # s0*r4
                  # s1 * RAYS
                  StateFeature(exps=[0, 1, 1, 0, 0, 0, 0]), # s1*r0 
                  StateFeature(exps=[0, 1, 0, 1, 0, 0, 0]), # s1*r1
                  StateFeature(exps=[0, 1, 0, 0, 1, 0, 0]), # s1*r2 
                  StateFeature(exps=[0, 1, 0, 0, 0, 1, 0]), # s1*r3
                  StateFeature(exps=[0, 1, 0, 0, 0, 0, 1])] # s1*r4


# Obtains x(s,a) (feature vector values) for a specific state, action
# s - current state
# a - value for each action variable (e.g. [0, 1, 0] for go forward)
def get_feature_vectors(s):
    policy_vector_dict = {}
    t_value_vector = []
    for state_feature in STATE_FEATURES:
        feature_value, policy_vec_dict = state_feature.eval(s)
        t_value_vector.append(feature_value)
        for action in ACTIONS:
            if action not in policy_vector_dict:
                policy_vector_dict[action] = []
            policy_vector_dict[action].extend(policy_vec_dict[action])
    return policy_vector_dict, np.array(t_value_vector)


# h(s, a, theta) = theta^T * X(s, a) in algorithm
def numerical_preference_h(action, theta, policy_vec):
    policy_vec_for_action = policy_vec[action]
    dot_prod = np.dot(theta, policy_vec_for_action)
    return dot_prod


def get_preferences(theta, policy_vec):
    # Normalize preferences so they are from -max to 0, to avoid overflow when raising to power
    prefs = np.array([numerical_preference_h(action_i, theta, policy_vec) for action_i in ACTIONS])
    return prefs - np.max(prefs)


def softmax(vec):
    exps = np.exp(vec) #vec should be <0 ovoid overflow (still results in same output due to maths)
    return exps / np.sum(exps)


def get_action_from_policy(theta, policy_vec):
    prefs = get_preferences(theta, policy_vec)
    policy_probs = softmax(prefs)
    chosen_action = random.choices(population=ACTIONS, weights=policy_probs)[0]
    return chosen_action, policy_probs, prefs


def my_grad(action, policy_vector, preferences):
    # x(s,a) - (Sum(x(s,ai)*e**pref)/Sum(e**pref))
    cur_policy_vals = np.array(policy_vector[action])
    exps = np.exp(preferences)
    multiplications = []
    for index, action_i in enumerate(ACTIONS):
        multiplication = np.array(policy_vector[action_i]) * exps[index]
        multiplications.append(multiplication)
    npmult = np.array(multiplications)
    numerator = np.sum(npmult, axis=0)
    denominator = np.sum(exps)
    result = cur_policy_vals - numerator/denominator
    return result

# def softmax_grad(softmax, gradient):
#     SM = softmax.reshape((-1, 1))
#     jac = np.diagflat(softmax) - np.dot(SM, SM.T)
#     print("hi")

# v(s,w) = w^T * X(s) // transpose of weight vector * policy feature vector
def value(weights, v_vector):
    return np.dot(weights, v_vector)


def grad_policy(action, policy_vector, policy_probs):
    policy_vals = [np.array(policy_vector[action_i]) for action_i in ACTIONS]
    expected_values = [prob*policy_vals[i] for i, prob in enumerate(policy_probs)]
    summed_array = sum(expected_values)
    # for val in expected_values[1:]:
    #     summed_array += val
    gradient = np.array(policy_vector[action]) - summed_array
    return gradient


# Generates binary tuple of size {size} from input {n}
# Note: Output is already reversed
def to_binary_tuple(n, size):
    ret = [0] * size
    for i in range(size):
        ret[size - (i+1)] = n % 2
        n = int(n/2)
    return ret


# Generates all possible action tuples from multi-discrete space
# TODO make modular, generators are hard
def gen_action_values(action_space_values):
    global ACTIONS
    if ACTIONS is None:
        ACTIONS = []
        for action1 in action_space_values[0]:
            for action2 in action_space_values[1]:
                this_tuple = (action1, action2)
                ACTIONS.append(this_tuple)
    return ACTIONS


# Given specific action tuple, assigns a value to each action using the 
# function callbacks in action_evals
def evaluate_actions(action_value_tuple, action_evals):
    evaluated_actions = []
    for index, action in enumerate(action_value_tuple):
        action_eval = action_evals[index](action)
        evaluated_actions.append(action_eval)
    return evaluated_actions

# Generate every combination of active/ inactive action
def gen_action_combos():
    global env, ACTION_COMBOS, COMBO_DIMENSION
    if ACTION_COMBOS is None:
        action_space = env.action_space
        action_dimension = len(action_space.nvec)
        action_values = gen_action_values(env.action_space_values)
        COMBO_DIMENSION = 2**action_dimension
        ACTION_COMBOS = [to_binary_tuple(i, action_dimension) for i in range(COMBO_DIMENSION)]



# TODO Non-zero weights?
def init_policy_weights(length):
    return np.zeros(length)


def init_value_weights(length):
    return np.zeros(length)


def init_weight_vectors(policy_len, value_len, overwrite = False):
    if not os.path.exists(FILE_NAME_POLICY_WEIGHTS) or overwrite:
        policy_weights = init_policy_weights(policy_len)
    else:
        policy_weights = np.loadtxt(FILE_NAME_POLICY_WEIGHTS, dtype=float)
        if len(policy_weights) != policy_len:
            raise ValueError(f"File {FILE_NAME_POLICY_WEIGHTS} loaded with incorrect length for policy weights.")
        if np.isnan(policy_weights).any():
            policy_weights = init_policy_weights(policy_len)
    if not os.path.exists(FILE_NAME_VALUE_WEIGHTS) or overwrite:
        value_weights = init_value_weights(value_len)
    else:
        value_weights = np.loadtxt(FILE_NAME_VALUE_WEIGHTS, dtype=float)
        if len(value_weights) != value_len:
            raise ValueError(f"File {FILE_NAME_VALUE_WEIGHTS} loaded with incorrect length for value weights.")
        if np.isnan(value_weights).any():
            value_weights = init_value_weights(value_len)
    return policy_weights, value_weights


def normalize(state):
    low = np.array([MIN_SPEED] + [STEER_MIN] + [0.] * NUM_SENSORS)
    high = np.array([MAX_SPEED] + [STEER_MAX] + [RAY_CAST_DISTANCE] * NUM_SENSORS)
    for i in range(len(low)):
        state_range = (high[i]-low[i])
        state[i] = (state[i]-low[i])/state_range
    return tuple(state)


def do_actor_critic_episode(theta_weights, value_weights, show_graphics=False, episode=None):
    # Init episode fields
    global do_terminate
    do_terminate = False
    current_state = normalize(list(env.reset()))
    gamma = DEFAULT_GAMMA
    learning_rate_value_weights = VALUE_LEARNING_RATE  # Note: alpha_w
    learning_rate_policy_weights = POLICY_LEARNING_RATE # Note: alpha_theta
    gamma_accumulator = 1  # Note: "I" in algorithm
    iteration = 1
    value_vector = None
    policy_features_dict = None
    tile_visited_count = 0
    while not do_terminate and iteration < EPISODE_MAX_ITERATIONS:
        rendering = False
        if show_graphics and episode%SHOW_GRAPHICS_INTERVAL==0:
            env.render()
            rendering=True
        if value_vector is None and policy_features_dict is None:
            (policy_features_dict, value_vector) = get_feature_vectors(current_state)

        selected_action, policy_probs, preferences = get_action_from_policy(theta_weights, policy_features_dict)

        if rendering and iteration % SHOW_ACTIONS_INTERVAL == 0 and SHOW_ACTIONS:
            print(selected_action)

        next_state, reward, do_terminate, tile_visited_count = env.step(selected_action)
        next_state = normalize(list(next_state))

        if do_terminate:
            gamma = 0

        (policy_features_dict, next_value_vector) = get_feature_vectors(next_state)
        cur_val = value(value_weights, value_vector)
        next_val = value(value_weights, next_value_vector)


        dell = reward + gamma * next_val - cur_val
        value_weights += learning_rate_value_weights * dell * np.array(value_vector)  # TODO check if this should be dell * x
        
        # policy_gradient = grad_policy(selected_action, policy_features_dict, policy_probs)
        my_gradient = grad_policy(selected_action, policy_features_dict, preferences)
        theta_weights += learning_rate_policy_weights * gamma_accumulator * dell * my_gradient

        gamma_accumulator *= gamma
        current_state = next_state
        value_vector = next_value_vector
        iteration += 1
        # End Actor-Critic iteration
    return theta_weights, value_weights, tile_visited_count
    # End Actor-Critic episode


# Main code
do_terminate = False
if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "human":
            # Human mode
            actions = np.array([0.0, 0.0, 0.0])
            setup_controls()
            while True:
                tiles_visited_list, tiles_per_iter = do_human(EPISODE_MAX_ITERATIONS, actions)
                do_terminate = False
                print(f"Human episode complete. Iter:{len(tiles_visited_list)}, Tiles per iter: {tiles_per_iter:.4f}, Tiles:{tiles_visited_list[-1]}")
        else:  # Actor-Critic Mode
            # Generate action table
            gen_action_combos()
            # Load weight vectors
            policy_length = len(STATE_FEATURES)*COMBO_DIMENSION
            value_length = len(STATE_FEATURES)
            try:
                (theta_weights, value_weights) = init_weight_vectors(policy_length, value_length)
            except ValueError as ve:
                print(ve)
                print("Overwriting files...")
                (theta_weights, value_weights) = init_weight_vectors(policy_length, value_length, overwrite=True)
            for episode_num in range(MAX_EPISODES):
                theta_weights, value_weights, tiles_visited = do_actor_critic_episode(theta_weights=theta_weights,
                                                                                      value_weights=value_weights,
                                                                                      show_graphics=SHOW_GRAPHICS,
                                                                                      episode=episode_num)
                print(f"Episode {episode_num}: {tiles_visited} tiles")
                # if episode_num % 100 == 0:
                np.savetxt(FILE_NAME_POLICY_WEIGHTS, theta_weights)
                np.savetxt(FILE_NAME_VALUE_WEIGHTS, value_weights)
            # End Actor-Critic "else" block
