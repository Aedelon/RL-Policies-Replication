import typing as tt
import gymnasium as gym
from collections import defaultdict, Counter
from torch.utils.tensorboard import SummaryWriter

GAMMA = 0.9

State = int
Action = int
RewardKey = tt.Tuple[State, Action, State]
TransitionKey = tt.Tuple[State, Action]

class Agent:
    def __init__(self, env: gym.Env):
        self.env = env
        self.state, _ = self.env.reset()
        self.rewards: tt.Dict[RewardKey, float] = defaultdict(float)
        self.transitions: tt.Dict[TransitionKey, Counter] = defaultdict(Counter)
        self.values: tt.Dict[State, float] = defaultdict(float)

    def play_n_random_steps(self, n: int):
        for _ in range(n):
            action = self.env.action_space.sample()
            new_state, reward, done, trunc, _ = self.env.step(action)
            reward_key = (self.state, action, new_state)
            self.rewards[reward_key] = float(reward)
            transition_key = (self.state, action)
            self.transitions[transition_key][new_state] += 1
            if done or trunc:
                self.state, _ = self.env.reset()
            else:
                self.state = new_state

    def calc_action_value(self, state: State, action: Action, gamma: float) -> float:
        target_counts = self.transitions[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for target_state, count in target_counts.items():
            reward_key = (state, action, target_state)
            reward = self.rewards[reward_key]
            value = reward + gamma * self.values[target_state]
            action_value += (count / total) * value
        return action_value

    def select_action(self, state: State, gamma: float) -> Action:
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action, gamma)
            if best_value is None or action_value > best_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env: gym.Env, gamma: float) -> float:
        total_reward = 0.0
        state, _ = env.reset()
        while True:
            action = self.select_action(state, gamma)
            new_state, reward, done, trunc, _ = env.step(action)
            reward_key = (state, action, new_state)
            self.rewards[reward_key] = float(reward)
            transition_key = (state, action)
            self.transitions[transition_key][new_state] += 1
            total_reward += reward
            if done or trunc:
                break
            state = new_state
        return total_reward

    def value_iteration(self, gamma: float):
        for state in range(self.env.observation_space.n):
            state_values = [
                self.calc_action_value(state, action, gamma) for action in range(self.env.action_space.n)
            ]
            self.values[state] = max(state_values)


class QAgent(Agent):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def value_iteration(self, gamma: float):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = 0.0
                target_counts = self.transitions[(state, action)]
                total = sum(target_counts.values())
                for target_state, count in target_counts.items():
                    reward_key = (state, action, target_state)
                    reward = self.rewards[reward_key]
                    best_action = self.select_action(state, gamma)
                    value = reward + gamma * self.values[(target_state, best_action)]
                    action_value += (count / total) * value
                self.values[(state, action)] = action_value

    def select_action(self, state: State, gamma: float) -> Action:
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or action_value > best_value:
                best_value = action_value
                best_action = action
        return best_action

def train(env: gym.Env, agent: Agent, test_episodes: int, gamma: float = 0.99):
    writer = SummaryWriter()

    iteration = 0
    best_reward = 0.0
    while True:
        iteration += 1
        agent.play_n_random_steps(1000)
        agent.value_iteration(gamma)

        reward = 0.0
        for _ in range(test_episodes):
            reward += agent.play_episode(env, gamma)

        reward /= test_episodes
        writer.add_scalar('reward', reward, iteration)
        if reward > best_reward:
            print("Best reward: %.2f" % reward)
            best_reward = reward
            print(f"{iteration}: best reward updated: {best_reward:.3} -> {reward:.3}")
            best_reward = reward
        if reward > 0.8:
            print("Solved in %d iterations!" % iteration)
            break

    writer.close()