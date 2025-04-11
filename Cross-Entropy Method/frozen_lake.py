from cross_entropy_method import *
import gymnasium as gym


class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        shape = (env.observation_space.n, )
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res


if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='ansi')
    env = DiscreteOneHotWrapper(env)
    train(env, batch_size=128, full_batch_size=500, learning_rate=0.001, gamma=0.9, solved_reward=0.75)