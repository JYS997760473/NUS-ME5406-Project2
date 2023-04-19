import torch
import os
import argparse
import joblib
import json
import time
from src.utilities import plot_test

def load_model(exp_name: str, deterministic=False):
    """ 
    load a pre-trained SAC model and environment
    """
    
    exp_dir = os.path.join(os.getcwd(), "model", exp_name)
    model_name = os.path.join(exp_dir, "model.pt")

    print(f"loading pre-trained model: {model_name} ...")
    model = torch.load(model_name)

    env_name = os.path.join(exp_dir, "vars.pkl")
    print(f"loading pre-trained environment: {env_name} ...")
    state = joblib.load(env_name)
    env = state['env']

    # make function for producing an action given a single state
    def action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            a = model.act(x)
        return a

    return action, env

def run(exp_name: str, env, action, episodes: int, seed: int=-1, render: bool=
        True):
    """
    run the pre-trained model in the environment
    test.json file will be created 
    """
    json_list = []
    start_time = time.time()
    total_success = 0
    for i in range(episodes):
        if seed == -1:
            obs, info = env.reset()
        else:
            obs, info = env.reset(seed = seed)
        done = False
        episode_reward = 0
        success = False
        steps = 0
        while done == False:
            if render == True:
                env.render()
            act = action(obs)
            obs, reward, terminated, truncated, info = env.step(act)
            if reward == -100:
                reward = -1
            reward = reward * 10
            done = terminated or truncated
            episode_reward += reward
            steps += 1
        if terminated == True and truncated == False and reward > 0:
            success = True
            total_success += 1
        print(f"episode:{i}, total reward:{episode_reward}, success:{success}")
        one_episode_dict = {'episode':i, 'steps': steps, 'success': success, 
                            'total_success': total_success,
                            'reward': episode_reward, 'time': time.time()-
                            start_time}
        json_list.append(one_episode_dict)
    # end all episodes, record output
    exp_dir = os.path.join(os.getcwd(), "model", exp_name)
    json_dict = {'episodes': json_list}
    json_file = os.path.join(exp_dir, 'test_output.json')
    file = open(json_file, 'w')
    json.dump(json_dict, file, indent=4)
    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='BipedalWalkerSAC', 
                            help='name of current experiment')
    parser.add_argument('--episode', type=int, default=10, 
                        help='number of episodes want to test')
    parser.add_argument('--seed', type=int, default=-1, 
                        help='environment seed')
    parser.add_argument('--render', type=bool, default=True, 
                        help='whether render the environment in testing')
    args = parser.parse_args()
    action, env = load_model(exp_name=args.exp_name)
    run(exp_name=args.exp_name, env=env, action=action, episodes=args.episode, 
        seed=args.seed, render=args.render)
    plot_test(exp_name=args.exp_name)