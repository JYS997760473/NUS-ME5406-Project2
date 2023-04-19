import argparse
import gym
import torch
from src.sac import SAC

def train(args: argparse.Namespace):
    """
    
    """
    environment = 'BipedalWalkerHardcore-v3'
    print(f"making environmnet "+environment)
    env = gym.make(environment, render_mode='human')
    # torch.set_num_threads(torch.get_num_threads())
    sac = SAC(env=env, exp_name=args.exp_name, num_episode=args.episode, 
              batch_size=args.batch_size, seed=args.seed)
    sac.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='BipedalWalkerSAC', 
                        help='name of current experiment')
    parser.add_argument('--gamma', type=float, default=0.99, 
                        help='gamma in RL algorithm')
    parser.add_argument('--episode', type=int, default=500, 
                        help='number of episodes in this experiment')
    parser.add_argument('--render', type=bool, default=True, 
                        help='whether render the environment during training')
    parser.add_argument('--batch_size', type=int, default=100, 
                        help='batch size in FIFO buffer replay experience')
    parser.add_argument('--seed', type=int, default=0, 
                        help='environment seed')
    args = parser.parse_args()

    print(f"Hyperparameters:{args}")
    train(args=args) 