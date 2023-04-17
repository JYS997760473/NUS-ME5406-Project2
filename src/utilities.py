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
    plt.figure()
    plt.plot(x, reward, label='Reward')
    plt.plot(x, average_reward, label='Average Reward')
    plt.legend()
    plt.title("Reward and Average Reward Collected During Training")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()

def plot_success(exp_name: str):
    """
    plot number of success episodes
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
            # print(f"loading {file_name} ...")
            pass
        else :
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 
                                    file_name)
        
        # load the output.json file
        json_file = open(file_name, 'r')
        data = json.loads(json_file.read())

        return data

    # load json file and get dict data
    data = get_data(exp_name=exp_name)["episodes"]

    # collect success number 
    episode_num = len(data)
    x = np.arange(0, episode_num)
    success = np.zeros((episode_num))
    for i in range(episode_num):
        if "success_num" in data[i].keys():
            # new version json file
            success[i] = data[i]["success_num"]
        else:
            # old version json file
            success[i] = data[i]["success"]

    # set seaborn style
    sns.set() 
    plt.figure()
    plt.plot(x, success, label='Number of Successful Episodes')
    plt.legend()
    plt.title("Number of Successful Episodes During Training")
    plt.xlabel("Episode")
    plt.ylabel("Number of Successful Episodes")
    plt.show()

def plot_time(exp_name: str):
    """
    plot time consumption
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
            # print(f"loading {file_name} ...")
            pass
        else :
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 
                                    file_name)
        
        # load the output.json file
        json_file = open(file_name, 'r')
        data = json.loads(json_file.read())

        return data

    # load json file and get dict data
    data = get_data(exp_name=exp_name)["episodes"]

    # collect success number 
    episode_num = len(data)
    x = np.arange(0, episode_num)
    total_time_np = np.zeros((episode_num))
    time_np = total_time_np.copy()
    time_average_np = time_np.copy()
    episode_stop = 50
    for i in range(episode_num):
        total_time_np[i] = data[i]["total_time"]
        time_np[i] = data[i]["time"]
        if i >= episode_stop - 1:
            time_average_np[i] = np.mean(time_np[i - episode_stop + 1: i])

    # set seaborn style
    sns.set() 
    plt.figure()
    plt.plot(x, total_time_np, label='Total Time Consumption')
    plt.legend()
    plt.title("Time Consumption During Training")
    plt.xlabel("Episode")
    plt.ylabel("Time (second)")
    plt.show()

    plt.figure()
    plt.bar(x=x, height=time_np, alpha=0.6, width = 0.8, 
            facecolor = 'darkblue', edgecolor = 'darkblue', lw=1, 
                label='Time of Every Episode')
    plt.plot(x, time_average_np, color="red", label=
                "Average Time of Every Episode")
    plt.title("Time Consumption of Every Episode During Training")
    plt.xlabel("Episode")
    plt.ylabel("Time (second)")
    plt.legend()
    plt.show()