import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm

from gymconnectx.envs import ConnectGameEnv
from qlearning.static_scenario.connect4_6x7.scenario_connected4_6x7 import Scenario_Connected4_6x7


def trainConnectX(q_net,
                  q_target,
                  memory,
                  optimizer,
                  batch_size,
                  gamma):
    # ! We sample from the same Replay Buffer n=10 times
    for _ in range(10):
        # ! Monte Carlo sampling of a batch
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        # ! Get the Q-values
        q_out = q_net(s)

        # ! DQN update rule
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class ReplayBufferConnectGame():
    def __init__(self, buffer_limit):
        self.buffer = deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class QConnectNet(nn.Module):
    def __init__(self, no_actions, no_states):
        super(QConnectNet, self).__init__()
        self.fc1 = nn.Linear(no_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, no_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, state, epsilon, possible_actions):
        a = self.forward(state)
        if random.random() < epsilon:
            return random.choice(possible_actions)
        else:
            mask = torch.full(a.size(), float('-inf'))
            mask[possible_actions] = 0
            a = a + mask
            return a.argmax().item()


def env_3x3():
    env = ConnectGameEnv(
        connect=3,
        width=3,
        height=3,
        reward_winner=3,
        reward_loser=-3,
        reward_draw=1,
        reward_hell=-0.5,
        reward_hell_prob=-1.5,
        reward_win_prob=1.5,
        reward_living=-0.1,
        obs_number=True,
    )
    return env


def env_connected4_6x7():
    env = ConnectGameEnv(
        connect=4,
        width=7,
        height=6,
        reward_winner=3,
        reward_loser=-3,
        reward_draw=1,
        reward_hell=-0.5,
        reward_hell_prob=-1.5,
        reward_win_prob=1.5,
        reward_living=-0.1,
        obs_number=True)
    return env


def run():
    env = env_connected4_6x7()
    no_actions = env.width
    no_states = env.width * env.height
    batch_size = 32
    gamma = 0.98

    buffer_limit = 50_000
    learning_rate = 0.005

    q_net = QConnectNet(no_actions=no_actions, no_states=no_states)
    q_target = QConnectNet(no_actions=no_actions, no_states=no_states)
    q_target.load_state_dict(q_net.state_dict())

    memory = ReplayBufferConnectGame(buffer_limit=buffer_limit)

    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)

    rewards_plot_player1 = []
    rewards_plot_player2 = []
    episode_reward_player_1 = 0.0
    episode_reward_player_2 = 0.0

    num_episodes = 100
    num_games = 100

    scenario = Scenario_Connected4_6x7()
    valid_scenario_train = [scenario.generate_permutations()[0]]

    for i in range(len(valid_scenario_train)):
        scenario_player_1_and_2 = valid_scenario_train[i]
        for n_episode in tqdm(range(num_episodes), desc="Training Episodes"):
            epsilon = max(0.01, 0.1 - 0.005 * (n_episode / 100))
            for game in tqdm(range(num_games), desc="Games per Episode", leave=False):
                last_state = env.reset()
                last_action = -1
                last_player = ""
                while not env.is_done:
                    try:
                        current_player = 'player_1' if env.get_current_player() == 1 else 'player_2'
                        state = env.get_obs()
                        # action = q_net.sample_action(torch.from_numpy(state).float(), epsilon, env.get_moves())
                        if env.current_step < 6:
                            action = scenario_player_1_and_2[env.current_step]  # based on scenario
                        else:
                            if current_player == 'player_1':
                                action = q_net.sample_action(torch.from_numpy(state).float(), epsilon, env.get_moves())
                            else:
                                action = env.get_action_random()

                        next_state, rewards, done, _, info = env.step(action)
                        reward = rewards[current_player]

                        done_mask = 0.0 if done else 1.0
                        memory.put((state, action, reward, next_state, done_mask))

                        if current_player == 'player_1':
                            episode_reward_player_1 += reward
                        else:
                            episode_reward_player_2 += reward

                        if done:
                            reward = rewards[last_player]
                            memory.put((last_state, last_action, reward, state, done_mask))

                            trainConnectX(q_net, q_target, memory, optimizer, batch_size, gamma)
                            break
                        else:
                            last_player = current_player
                            last_state = state
                            last_action = action
                            env.current_step += 1

                    except Exception as e:
                        print(f"An error occurred: {str(e)}")
                        break
            q_target.load_state_dict(q_net.state_dict())
            rewards_plot_player1.append(episode_reward_player_1)
            rewards_plot_player2.append(episode_reward_player_2)
            episode_reward_player_1 = 0.0
            episode_reward_player_2 = 0.0

    # Save the trained Q-net
    torch.save(q_net.state_dict(), "dqn.pth")

    # Plot the training curve
    plt.plot(rewards_plot_player1, label='Reward per Episode (Player 1)')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.legend()
    plt.savefig("training_curve_player_1.png")
    plt.show()

    plt.plot(rewards_plot_player2, label='Reward per Episode (Player 2)')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.legend()
    plt.savefig("training_curve_player_2.png")
    plt.show()


if __name__ == "__main__":
    run()
