import gymnasium as gym
from q_iteration_v_iteration_method import Agent, train, QAgent

ENV_NAME = 'FrozenLake-v1'
TEST_EPISODES = 20


def main():
    test_env = gym.make(ENV_NAME)
    agent = QAgent(gym.make(ENV_NAME))
    train(test_env, agent, TEST_EPISODES, gamma=0.99)



if __name__ == '__main__':
    main()
