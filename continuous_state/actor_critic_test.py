# Actor-Critic implementation
# Algorithm referenced from RL textbook p. 332 (PDF p. 354)
# This is being ran on a modified state representation (see car_racing_pos.py)
import sys
import numpy as np
import gym

"""
Learning Constants
"""

EPISODE_MAX_ITERATIONS = 5000    # Max number of iterations in one episode


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
    global do_terminate_qlearn
    # Perform a run using the human
    env.reset()
    iteration_ctr = 0

    tiles_visited_list_human = []
    tiles_per_iter_human = 0
    current_average_reward = 0

    # Repeat until max iterations have passed or agent reaches terminal state
    while not do_terminate_qlearn and iteration_ctr <= max_iterations:
        # Keep that UI train rolling
        env.render()

        # Perform action, update state
        next_state, reward, do_terminate_qlearn, tile_visited_count = env.step(actions_human)
        iteration_ctr += 1

        # Update reward trackers, only use tile visited so that the reward function can be evaluated independently
        tiles_visited_list_human.append(tile_visited_count)
        tiles_per_iter_human = tile_visited_count / iteration_ctr
        current_average_reward = (
            current_average_reward*(iteration_ctr-1) + reward)/iteration_ctr

        # After some number of iterations we automatically terminate the episode
        if iteration_ctr > max_iterations:
            # print("Note: Terminating episode due to max iterations exceeded")
            do_terminate_qlearn = True
    return tiles_visited_list_human, tiles_per_iter_human


# Main code
if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "human":
            # Human mode
            actions = np.array([0.0, 0.0, 0.0])
            setup_controls()
            while True:
                tiles_visited_list, tiles_per_iter = do_human(EPISODE_MAX_ITERATIONS, actions)
                do_terminate_qlearn = False
                print(f"Human episode complete. Iter:{len(tiles_visited_list)}, Tiles per iter: {tiles_per_iter:.4f}, Tiles:{tiles_visited_list[-1]}")
        else:
            # Actor-Critic Mode
            pass
