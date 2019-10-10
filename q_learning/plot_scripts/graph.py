import os
import numpy as np
import matplotlib.pyplot as plt

EXPERIMENT = "hyperparametersA"
EXPERIMENT_TITLE = "Double Expected Sarsa"

def get_list_of_rewards_files(rewards_directory, experiment_name, metric_name):
    cur_path = os.path.dirname(rewards_directory)
    reward_path = os.path.join(cur_path, f"{experiment_name}\\{metric_name}\\")
    files = os.listdir(reward_path)
    if metric_name == "tiles_per_iter":
        files = sorted(files, key = lambda file: int(file.split("_")[-6]))  # Sort based on episode name
    elif metric_name == "tiles_visited":
        files=sorted(files, key = lambda file: int(file.split("_")[-2][2:])) #Sort based on episode name

    return files


def get_data(rewards_directory, experiment_name, metric_name):
    # Returns list of tuples of x, y data
    files = get_list_of_rewards_files(rewards_directory, experiment_name, metric_name)
    experiment_path = os.path.join(rewards_directory, experiment_name)
    metric_path = os.path.join(experiment_path, metric_name)
    
    if metric_name == "tiles_per_iter":
        avgs = []
        for loc_file_path in files:
            path = os.path.join(metric_path, loc_file_path)
            data = np.loadtxt(path)
            vals = data.tolist()
            if isinstance(vals, list):
                avgs.extend(vals)
            else:
                avgs.append(vals)
        return [(range(len(avgs)), avgs)]
    elif metric_name == "tiles_visited":
        group = []
        for loc_file_path in files:
            path = os.path.join(metric_path, loc_file_path)
            data = np.loadtxt(path)
            vals = data.tolist()
            y = []
            if isinstance(vals, list):
                y.extend(vals)
            else:
                y.append(vals)
            group.append((range(len(y)), y))
        return group


def graph_data(rewards_directory, experiment_name, metric_name, filter_fn=lambda x: True):
    data = get_data(rewards_directory, experiment_name, metric_name)
    
    fig = plt.figure(0)
    fig.canvas.set_window_title('CS 4033/5033 - Cameron Bost, Bryce Hennen')
    num_datasets = len(data)
    color = (0.1, 0.2, 0.5, 0.9)
    color_incr = (0.9-0.1)/len(data)
    for dataset in data:
        color = (color[0] + color_incr, color[1], color[2], color[3])
        filtered_data= [[],[]]
        for index, y_val in enumerate(dataset[1]):
            if filter_fn(y_val):
                filtered_data[0].append(dataset[0][index])
                filtered_data[1].append(y_val)
        x, y = filtered_data
        x = np.asarray(x)
        y = np.asarray(y)
        plt.plot(x, y, color=color)
        if metric_name == "tiles_per_iter":
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            plt.plot(x, m*x + c, "r", label=f'Fitted line - Slope:{m:4f}, Intercept:{c:4f}')
            plt.legend()
    if metric_name == "tiles_per_iter":    
        plt.xlabel('Episode')
        plt.ylabel('Tiles Per Iteration')
    elif metric_name == "tiles_visited":
        plt.xlabel('Completed Iterations')
        plt.ylabel('Tiles')
    plt.title(EXPERIMENT_TITLE)
    plt.show()



if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    rewards_directory = os.path.join(cur_dir, f"..\\rewards\\")
    graph_data(rewards_directory, EXPERIMENT, "tiles_per_iter", filter_fn=lambda x: x < 0.5)
    graph_data(rewards_directory, EXPERIMENT, "tiles_visited")
