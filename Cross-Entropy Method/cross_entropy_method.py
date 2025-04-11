import typing as tt
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter

class Net(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, n_actions: int):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class EpisodeStep:
    obs: np.ndarray
    action: int


@dataclass
class Episode:
    reward: float
    steps: tt.List[EpisodeStep]


def iterate_batches(env: gym.Env, net: Net, batch_size: int) -> tt.Generator[tt.List[Episode], None, None]:
    """Yield batches of episodes from an environment by interacting with it using a neural network.

    Args:
        env: The Gymnasium environment to interact with.
        net: The neural network used to decide actions.
        batch_size: Number of episodes per batch.

    Yields:
        A list of Episodes, each containing the total reward and steps taken in that episode.

    This function continuously interacts with the environment, generating episodes by
    using the neural network to choose actions based on the current observation. Each time
    a batch of `batch_size` episodes is collected, it is yielded back.
    """
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs, _ = env.reset()
    sm = nn.Softmax(dim=1)

    while True:
        # Convert observation to tensor and get action probabilities
        obs_v = torch.tensor(obs, dtype=torch.float32)
        act_probs_v = sm(net(obs_v.unsqueeze(dim=0)))
        act_probs = act_probs_v.data.numpy()[0]

        # Choose action based on probabilities
        action = np.random.choice(act_probs.shape[0], p=act_probs)

        # Step through the environment and record the transition
        next_obs, reward, done, trunc, _ = env.step(action)
        episode_reward += float(reward)
        step = EpisodeStep(obs, action)
        episode_steps.append(step)

        # Check if the episode is complete (done or truncated)
        if done or trunc:
            e = Episode(episode_reward, episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []

            # Reset environment for the next episode
            obs, _ = env.reset()

            # Yield the batch when it reaches the desired size
            if len(batch) == batch_size:
                yield batch
                batch = []

        # Update observation for the next step
        obs = next_obs
        

def filter_batch(batch: tt.List[Episode], percentile: float, gamma: float) \
        -> tt.Tuple[tt.List[Episode], torch.FloatTensor, torch.LongTensor, float, float]:
    rewards = list(map(lambda s: s.reward * (gamma ** len(s.steps)), batch))
    reward_bound = float(np.percentile(rewards, percentile))
    reward_mean = float(np.mean(list(map(lambda s: s.reward, batch))))

    train_obs: tt.List[np.ndarray] = []
    train_actions: tt.List[int] = []
    elite_batch: tt.List[Episode] = []

    for episode, reward in zip(batch, rewards):
        if gamma < 1.0:
            if reward > reward_bound:
                train_obs.extend(map(lambda s: s.obs, episode.steps))
                train_actions.extend(map(lambda s: s.action, episode.steps))
                elite_batch.append(episode)
        else:
            if reward >= reward_bound:
                train_obs.extend(map(lambda s: s.obs, episode.steps))
                train_actions.extend(map(lambda s: s.action, episode.steps))
                elite_batch.append(episode)


    train_obs_v: torch.FloatTensor = torch.tensor(np.array(train_obs), dtype=torch.float32)
    train_actions_v: torch.LongTensor = torch.tensor(train_actions, dtype=torch.long)

    return elite_batch, train_obs_v, train_actions_v, reward_bound, reward_mean


def train(
        env: gym.Env,
        hidden_size: int = 256,
        batch_size: int = 32,
        full_batch_size: int = 128,
        learning_rate: float = 1e-3,
        percentile: float = 70,
        gamma: float = 0.99,
        solved_reward: float = 475,
):
    assert env.observation_space.shape is not None, 'The observation space of the environment is None'
    obs_size = env.observation_space.shape[0]
    assert isinstance(env.action_space, gym.spaces.Discrete), 'The action space of the environment is not discrete'
    n_actions = int(env.action_space.n)

    net = Net(obs_size, hidden_size, n_actions)
    print(net)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    writer = SummaryWriter()

    full_batch = []
    for iter_no, batch in enumerate(iterate_batches(env, net, batch_size)):
        full_batch, obs, acts, reward_bound, reward_mean = filter_batch(full_batch + batch, percentile, gamma)
        if not full_batch:
            continue
        full_batch = full_batch[-full_batch_size:]

        optimizer.zero_grad()
        action_scores = net(obs)
        loss = loss_fn(action_scores, acts)
        loss.backward()
        optimizer.step()

        print("%d: loss=%.3f, reward_mean=%.2f, reward_bound=%.2f, batch=%d"
              % (iter_no, loss.item(), reward_mean, reward_bound, len(full_batch)))
        writer.add_scalar('loss', loss.item(), iter_no)
        writer.add_scalar('reward_mean', reward_mean, iter_no)
        writer.add_scalar('reward_bound', reward_bound, iter_no)
        writer.add_scalar('batch', len(full_batch), iter_no)

        if reward_mean > solved_reward:
            print('Done')
            break

    writer.close()