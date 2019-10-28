import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

EXPERIMENT1 = "double_sarsa"
EXPERIMENT2 = "single_sarsa"
EXPERIMENT_TITLE = "Single Expected Sarsa vs Double Expected Sarsa"

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

def graph_data(rewards_directory, experiment_name, metric_name, start_color, end_color, filter_fn=lambda x: True):
    data = get_data(rewards_directory, experiment_name, metric_name)
    
    fig = plt.figure(0)
    fig.canvas.set_window_title('CS 4033/5033 - Cameron Bost, Bryce Hennen')
    num_datasets = len(data)
    color_incr = (end_color - start_color)/len(data)
    npcolor = start_color
    for episode, dataset in enumerate(data):
        if episode > 0: npcolor += color_incr
        color = list(np.asarray(npcolor))
        filtered_data= [[],[]]
        for index, y_val in enumerate(dataset[1]):
            if filter_fn(y_val):
                filtered_data[0].append(dataset[0][index])
                filtered_data[1].append(y_val)
        x, y = filtered_data
        x = np.asarray(x)
        y = np.asarray(y)
        if metric_name == "tiles_visited":
            if episode == 0:
                plt.plot(x, y, color=color, label = "Early Episodes")
            elif episode == len(data)-1:
                plt.plot(x, y, color=color, label = "Later Episodes")
            else:
                plt.plot(x, y, color=color)
            plt.legend()
        elif metric_name == "tiles_per_iter":
            plt.plot(x, y, color=color)
            # Lowess Smoothing Regression
            frac = 1.0/10
            fitted_vals = sm.nonparametric.lowess(y, x, frac = frac, return_sorted=False)
            plt.plot(x, fitted_vals, color=end_color, label=f"{experiment_name} - Lowess Regression")
            plt.legend()
            
            # # Rolling Average 
            # window = 50
            # weights = np.repeat(1.0, window)/window
            # avgs = np.convolve(weights, y, "valid")
            # plt.plot(range(len(avgs)), avgs, "r", label=f"Rolling Average - Window Size = {window}")
            # plt.legend()
            
            # Running Average
            # running_avg = 0
            # avgs = []
            # for index, datum in enumerate(y):
            #     total = index + 1
            #     running_avg = (running_avg*(total-1) + datum)/total
            #     avgs.append(running_avg)
            # plt.plot(x, avgs, "r")
            
            # Linear Regression
            # A = np.vstack([x, np.ones(len(x))]).T
            # m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            # plt.plot(x, m*x + c, "r", label=f'Fitted line - Slope:{m:4f}, Intercept:{c:4f}')
            # plt.legend()
            
    if metric_name == "tiles_per_iter":    
        plt.xlabel('Episode')
        plt.ylabel('Tiles Per Iteration')
    elif metric_name == "tiles_visited":
        plt.xlabel('Completed Iterations')
        plt.ylabel('Tiles')
    
def graph_finish():
    plt.title(EXPERIMENT_TITLE)
    plt.show()



if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    rewards_directory = os.path.join(cur_dir, f"qlearn_results\\")
    # graph_data(rewards_directory, EXPERIMENT1, "tiles_per_iter", filter_fn=lambda x: x < 0.5)
    # graph_data(rewards_directory, EXPERIMENT2, "tiles_visited")

    start_color1 = np.array((255, 100, 0, 50)) / 255.0
    end_color1 = np.array((255, 0, 0, 255)) / 255.0 
    #End color is the loess regression line color for tiles_per_iter, otherwise it is the last episodes tiles_visited line color

    start_color2 = np.array((0, 150, 200, 50)) / 255.0
    end_color2 = np.array((0, 0, 255, 255)) / 255.0

    graph_data(rewards_directory, EXPERIMENT1, "tiles_per_iter", start_color1, end_color1, filter_fn=lambda x: x < 0.5)
    graph_data(rewards_directory, EXPERIMENT2, "tiles_per_iter", start_color2, end_color2, filter_fn=lambda x: x < 0.5)
    graph_finish()
