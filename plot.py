from src.utilities import *
import argparse




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='4.17.12:36', 
                        help='name of current experiment')
    args = parser.parse_args()
    # plot reward
    plot_reward(exp_name=args.exp_name)

    # plot success
    plot_success(exp_name=args.exp_name)

    # plot time
    plot_time(exp_name=args.exp_name)