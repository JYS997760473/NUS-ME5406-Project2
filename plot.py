from src.utilities import *
import argparse

def plot(exp_name: str):
    """
    plot main function
    """
    # plot reward
    plot_reward(exp_name=exp_name)

    # plot success
    plot_success(exp_name=exp_name)

    # plot time
    plot_time(exp_name=exp_name)

    # plot steps
    plot_steps(exp_name=exp_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='4.17.12:36', 
                        help='name of current experiment')
    args = parser.parse_args()
    # # plot reward
    # plot_reward(exp_name=args.exp_name)

    # # plot success
    # plot_success(exp_name=args.exp_name)

    # # plot time
    # plot_time(exp_name=args.exp_name)

    # # plot steps
    # plot_steps(exp_name=args.exp_name)
    plot(exp_name=args.exp_name)