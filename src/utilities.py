import json
import seaborn as sns
import os
import errno
import matplotlib.pyplot as plt
import numpy as np

def get_data(exp_name: str) -> dict:
    """
    find json file and load the data
    """
    exp_dir = os.path.join(os.getcwd(), "model", exp_name)
    file_name = os.path.join(exp_dir, "output.json")

    # first check whether the experiment name is valid and the output.json file
    # exits
    if os.path.exists(file_name):
        print(f"valid experiment name")
    else :
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 
                                file_name)
    
    # load the output.json file
    json_file = open(file_name, 'r')
    data = json.loads(json_file.read())

    return data

def plot_reward(exp_name: str):
    """
    plot reward
    """
    def get_data(exp_name: str) -> dict:
        """
        find json file and load the data
        """
        exp_dir = os.path.join(os.getcwd(), "model", exp_name)
        file_name = os.path.join(exp_dir, "output.json")

        # first check whether the experiment name is valid and the output.json 
        # file exits
        if os.path.exists(file_name):
            print(f"loading {file_name} ...")
        else :
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 
                                    file_name)
        
        # load the output.json file
        json_file = open(file_name, 'r')
        data = json.loads(json_file.read())

        return data

    # load json file and get dict data
    data = get_data(exp_name=exp_name)["episodes"]

    # collect reward
    episode_num = len(data)
    x = np.arange(0, episode_num)
    reward = np.zeros(episode_num)
    average_reward = reward.copy()
    average_stop = 50
    for i in range(episode_num):
        reward[i] = data[i]['reward']
        if i >= average_stop - 1:
            average_reward[i] = np.mean(reward[i - average_stop + 1: i])

    # set seaborn style
    sns.set() 
    plt.plot(x, reward, label='reward')
    plt.plot(x, average_reward, label='average reward')
    plt.legend()
    plt.show()