import argparse









if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardcore', type=bool, default=True, help='hardcore or normal version')
    parser.add_argument('--exp_name', type=str, default='BipedalWalkerSAC', help='name of current experiment')
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma in RL algorithm')
    parser.add_argument('--episode', type=int, default=500, help='number of episodes in this experiment')
    parser.add_argument('--render', type=bool, default=True, help='whether render the environment during training')
    args = parser.parse_args()
    