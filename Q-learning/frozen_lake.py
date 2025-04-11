import gymnasium as gym
from q_learning import Agent, train

ENV_NAME = 'FrozenLake-v1'
TEST_EPISODES = 20


def main():
    test_env = gym.make(ENV_NAME)
    agent = Agent(gym.make(ENV_NAME), gamma=0.9, alpha=0.2)
    train(test_env, agent, TEST_EPISODES)



if __name__ == '__main__':
    main()
