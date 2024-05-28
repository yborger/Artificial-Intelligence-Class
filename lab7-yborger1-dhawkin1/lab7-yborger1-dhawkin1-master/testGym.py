"""
Use this program to try out different environments in gymnasium.
"""

from time import sleep
import gymnasium as gym
import numpy as np

# This is another environment you can try
env = gym.make("LunarLander-v2", render_mode="human")

#env = gym.make("CartPole-v1", render_mode="human")
print("observation space:", env.observation_space.shape)
print("action space:", env.action_space.n)

# Try the environment for several episodes
for i in range(2):
    print("-"*70)
    print("Episode", i)
    state, _ = env.reset() # returns the initial state 
    total_reward = 0
    for j in range(200):
        env.render()
        sleep(0.05) # used to slow down animation
        action = env.action_space.sample() # choose a random action
        result = env.step(action)
        # Step returns 5 items, but we only care about the first 3
        next_state, reward, done, _, _ = result
        total_reward += reward
        print("action", action, "reward:", reward)
        if done:
            break
    print("Episode ended after", j, "steps, total reward:", total_reward )

env.close()

