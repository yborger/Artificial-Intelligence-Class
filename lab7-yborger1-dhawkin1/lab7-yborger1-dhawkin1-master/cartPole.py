import gymnasium as gym
from deepQCart import *
import sys

def check_command_line():
    """
    Verify that the correct command-line arguments have been provided.
    If not, print a usage message and exit the program.
    """
    args = sys.argv
    valid = True
    if len(args) != 3:
        valid = False
    elif not (args[1] == "train" or args[1] == "test"):
        valid = False
    if not valid:
        print("Invalid command-line arguments")
        print("Usage: train numEpisodes or test wtsFile")
        exit()

def main():
    """
    For cart pole problem, allow 200 steps per episode.
    Batchsize is 128 and iterations per batch is 8.
    """
    check_command_line()
    if sys.argv[1] == "train":
        env = gym.make("CartPole-v1")
        agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
        agent.train(env, int(sys.argv[2]), 200, 128, 8, "CartPole")
        agent.plot_history()
    elif sys.argv[1] == "test":
        env = gym.make("CartPole-v1", render_mode="human")
        agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
        try:
            agent.load_weights(sys.argv[2])
        except:
            print("Unable to load weights from file:", sys.argv[2])
        agent.test(env, 3, 200)
    env.close()

main()
    
