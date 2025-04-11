from cross_entropy_method import *
import gymnasium as gym


if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = gym.wrappers.RecordVideo(env, video_folder='./Cross-Entropy Method/CartPole-v1_videos')
    train(env, learning_rate=0.01, gamma=1.0)