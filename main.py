import argparse
import gym
from src.sac import SAC

def main(args: argparse.Namespace):
    """
    
    """
    environment = 'BipedalWalkerHardcore-v3' if args.hardcore == True else 'BipedalWalker-v3'
    env = gym.make(environment, render_mode='human')
    print(f"environmnet is "+environment)
    SAC(env=env, num_episode=1500)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardcore', type=bool, default=True, help='hardcore or normal version')
    parser.add_argument('--exp_name', type=str, default='BipedalWalkerSAC', help='name of current experiment')
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma in RL algorithm')
    parser.add_argument('--episode', type=int, default=500, help='number of episodes in this experiment')
    parser.add_argument('--render', type=bool, default=True, help='whether render the environment during training')
    args = parser.parse_args()

    main(args=args)
    