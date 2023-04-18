import torch
import os
import argparse
import joblib

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

def run(env, action, episodes: int):
    """
    run the pre-trained model in the environment
    """

    for i in range(episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        success = False
        while done == False:
            env.render()
            act = action(obs)
            obs, reward, terminated, truncated, info = env.step(act)
            if reward == -100:
                reward = -1
            reward = reward * 10
            done = terminated or truncated
            episode_reward += reward
        success = terminated == True and truncated == False and reward > 0
        print(f"episode:{i}, total reward:{episode_reward}, success:{success}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='BipedalWalkerSAC', 
                            help='name of current experiment')
    parser.add_argument('--episode', type=int, default=10, 
                        help='number of episodes want to test')
    args = parser.parse_args()
    action, env = load_model(exp_name=args.exp_name)
    run(env=env, action=action, episodes=args.episode)