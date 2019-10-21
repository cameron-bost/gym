# Actor-Critic implementation
# Algorithm referenced from RL textbook p. 332 (PDF p. 354)
# This is being ran on a modified state representation (see car_racing_pos.py)
import os
import sys
import numpy as np
import gym

"""
Learning Parameters
"""
MAX_EPISODES = 20000
EPISODE_MAX_ITERATIONS = 5000    # Max number of iterations in one episode
DEFAULT_GAMMA = 0.9

"""
Model Constants
"""
ACTIONS = None
ACTION_COMBO_COEFFICIENTS = None

"""
Results Constants
"""
FILE_NAME_VALUE_WEIGHTS = "value_weights.gz"
FILE_NAME_POLICY_WEIGHTS = "policy_weights.gz"
DIRECTORY_WEIGHT_BACKUP = "weight_backups"

# Gym instance
env = gym.make('CarRacing5033ContinuousState-v0')


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


# A feature X that receives state as input when evaluating.
# Upon evaluation, this will return the state feature value (used as x(s)) as well as all possible policy feature
# values.
class StateFeature:
    def __init__(self, exps):
        self.exps = exps
        self.state_value = 0
        self.action_values = {}

    def eval(self, state):
        self.state_value = 0  # TODO product(state_i ^ exps_i for i)
        self.action_values = {}
        for a in ACTIONS:
            action_combo_values = []
            for combo in ACTION_COMBO_COEFFICIENTS:
                action_combo_value = np.prod([a[i]*combo[i] for i in range(len(combo))])
                action_combo_values.append(action_combo_value)
            self.action_values[a] = action_combo_values
        return self.state_value, self.action_values

    def eval_action(self, a):
        return self.action_values[a]


# Obtains x(s,a) - feature vector
# s - current state
# a - value for each action variable (e.g. [0, 1, 0] for go forward)
def get_feature_vectors(s, a):
    # TODO state_action_tuple = s+a
    # for each feature f in policy_features:
    #   policy_vector.append(f.eval(state_action_tuple))
    # for each feature f in value_features:
    #   value_vector.append(f.eval(s))
    # return (policy_vector, value_vector)
    pass


def numerical_preference_h(state, action, theta, policy_vector):
    # TODO np.dot(theta, get_feature_vector(state, action))
    pass


def policy(action, state, theta, policy_vector):
    # TODO numerator = e^numerical_preference_h(state, action, theta)
    # denom = sum([e^numerical_preference_h(state, action_i, theta) for action_i in ACTIONS])
    pass


def get_action_from_policy(current_state, theta, policy_vector):
    # TODO choose_action([policy(state, action_i, theta) for action_i in ACTIONS])
    pass


def value(state, weights, value_vector):
    # TODO v(s,w) = w^T * X(s) // transpose of weight vector * policy feature vector
    pass


def grad_policy(action, state, theta, policy_vector):
    # TODO x(s,a) - sum([policy(action_i, state, theta)*policy_features(state, action_i) for action_i in ACTIONS])
    pass


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
                for action3 in action_space_values[2]:
                    this_tuple = [action1, action2, action3]
                    ACTIONS.append(this_tuple)
    return ACTIONS


def gen_action_coefficients():
    global env, ACTIONS, ACTION_COMBO_COEFFICIENTS
    if ACTION_COMBO_COEFFICIENTS is None:
        action_n = int(np.prod(env.action_space.nvec))
        action_space = env.action_space
        action_dimension = len(action_space.nvec)
        action_values = gen_action_values(env.action_space_values)
        combo_dimension = 2**action_dimension
        ACTION_COMBO_COEFFICIENTS = []
        combos = [to_binary_tuple(i, action_dimension) for i in range(combo_dimension)]
        for action_tuple_idx in range(action_n):
            action_value_tuple = action_values[action_tuple_idx]
            action_value_coefficient_set = []
            for combo_idx in range(combo_dimension):
                combo = combos[combo_idx]
                action_combo_value = 1
                for action_idx in range(action_dimension):
                    a = action_value_tuple[action_idx]
                    e = combo[action_idx]
                    action_combo_value *= a**e
                action_value_coefficient_set.append(action_combo_value)
            ACTION_COMBO_COEFFICIENTS.append(action_value_coefficient_set)


def init_policy_weights():
    pass


def init_value_weights():
    pass


def init_weight_vectors():
    if not os.path.exists(FILE_NAME_POLICY_WEIGHTS):
        policy_weights = init_policy_weights()
    else:
        policy_weights = np.loadtxt(FILE_NAME_POLICY_WEIGHTS, dtype=float)
    if not os.path.exists(FILE_NAME_VALUE_WEIGHTS):
        value_weights = init_value_weights()
    else:
        value_weights = np.loadtxt(FILE_NAME_VALUE_WEIGHTS, dtype=float)
    return policy_weights, value_weights


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
            gen_action_coefficients()
            # Load weight vectors
            (theta_weights, value_weights) = init_weight_vectors()
            for episode_num in range(MAX_EPISODES):
                # Init episode fields
                current_state = env.reset()
                gamma = DEFAULT_GAMMA
                learning_rate_value_weights = 0     # Note: alpha_w
                learning_rate_policy_weights = 0    # Note: alpha_theta
                gamma_accumulator = 1               # Note: "I" in algorithm
                iteration = 1
                while not do_terminate and iteration < EPISODE_MAX_ITERATIONS:
                    (policy_vector, value_vector) = get_feature_vectors(current_state)

                    selected_action = get_action_from_policy(current_state, theta_weights)

                    next_state, reward, do_terminate, tile_visited_count = env.step(selected_action)

                    if do_terminate:
                        gamma = 0
                    dell = reward + gamma*value(next_state, value_weights) - value(current_state, value_weights)

                    value_weights += learning_rate_value_weights * dell * value_vector  # TODO check if this should be dell * x

                    policy_gradient = grad_policy(selected_action, current_state, theta_weights)
                    theta_weights += learning_rate_policy_weights * gamma_accumulator * dell * policy_gradient

                    gamma_accumulator *= gamma
                    current_state = next_state
                    iteration += 1
                    # End Actor-Critic iteration
                # End Actor-Critic episode
            # End Actor-Critic "else" block
