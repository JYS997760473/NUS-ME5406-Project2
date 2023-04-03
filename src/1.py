import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="rgb_array")

observation, info = env.reset()

obs_space = env.observation_space
action_space = env.action_space
# for i in range(10):
#     env_screen = env.render()
#     plt.imshow(env_screen)
#     plt.show()
#     env.step(env.action_space.sample())

env.close()