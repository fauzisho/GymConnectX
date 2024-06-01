import os
import random
import csv

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from gymconnectx.envs import ConnectGameEnv


class QLearningAgent:
    def __init__(self, role='player_1', q_table={}, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.role = role  # Player role: 'player_1' or 'player_2'
        self.q_table = q_table  # Q-table initially empty
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def get_state_key(self, state):
        """Convert state to a string key for Q-table."""
        return str(state)

    def choose_action(self, state, possible_actions) -> int:
        """Choose an action using an epsilon-greedy policy."""
        state_key = self.get_state_key(state)
        if random.random() < self.epsilon:  # Exploration
            return random.choice(possible_actions)
        else:  # Exploitation
            if state_key in self.q_table and self.q_table[state_key]:
                possible_q_values = {action: self.q_table[state_key][action] for action in possible_actions if
                                     action in self.q_table[state_key]}
                if possible_q_values:
                    return max(possible_q_values, key=possible_q_values.get)
                else:
                    return random.choice(possible_actions)
            else:
                return random.choice(possible_actions)

    def update_q_value(self, state, action, reward, next_state):
        """Update the Q-value for a state-action pair."""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if action not in self.q_table[state_key]:
            self.q_table[state_key][action] = 0

        # Ensure next_state_key is in the Q-table; if not, use an empty dict
        next_q_values = self.q_table.get(next_state_key, {})

        # Get max Q-value for next state
        next_max = max(next_q_values.values(), default=0)
        self.q_table[state_key][action] += self.alpha * (
                reward + self.gamma * next_max - self.q_table[state_key][action])

    def save_q_table_to_csv(self, file_name="q_table_old.csv"):
        """Save the Q-table to a CSV file."""
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["state", "action", "q_value"])
            for state, actions in self.q_table.items():
                for action, q_value in actions.items():
                    writer.writerow([state, action, q_value])

    @staticmethod
    def load_q_table_from_csv(file_name="q_table_old.csv"):
        """Load the Q-table from a CSV file."""
        q_table = {}
        if os.path.exists(file_name):
            with open(file_name, 'r', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    state = row['state']
                    action = int(row['action'])
                    q_value = float(row['q_value'])
                    if state not in q_table:
                        q_table[state] = {}
                    q_table[state][action] = q_value
        return q_table


def train_agent_env(num_training):
    env = ConnectGameEnv(connect=3, width=3, height=3, reward_winner=3, reward_loser=-3, living_reward=-0.1)
    roles = ['player_1', 'player_2']
    agent_1 = QLearningAgent(roles[0])
    agent_2 = QLearningAgent(roles[1])
    last_state = {role: None for role in roles}
    last_action = {role: None for role in roles}

    for game in range(num_training):
        env.reset()
        done = False
        while not done:
            current_role = roles[env.get_current_player() - 1]
            if env.get_current_player() == 1:
                agent = agent_1
            else:
                agent = agent_2

            state = str(env.get_player_observations())
            possible_actions = env.get_moves()
            action = agent.choose_action(state, possible_actions)

            next_state, rewards, done, _, info = env.step(action)
            env.render("terminal_display")
            reward = rewards[current_role]

            # Store the last action and state
            last_state[current_role] = state
            last_action[current_role] = action

            agent.update_q_value(state, action, reward, str(next_state))

            if done:
                # Game over: Update for both players
                current_role = roles[env.get_current_player() - 1]
                last_reward = rewards[current_role]
                agent.update_q_value(last_state[current_role], last_action[current_role], last_reward, str(next_state))

                print("------------------------")
                env.render('terminal_display')
                print(env.get_game_status())
                print(f'game: {game}, action: {action}, reward: {reward}')
                print("------------------------")

        # Save the Q-table to CSV after training
        agent_1.save_q_table_to_csv(f'q_table_player_1.csv')
        agent_2.save_q_table_to_csv(f'q_table_player_2.csv')


def env_qtable():
    return ConnectGameEnv(
        connect=3,
        width=3,
        height=3,
        reward_winner=3,
        reward_loser=-3,
        living_reward=-0.1, )


def play_with_q_table(file_name="q_table_player_2.csv"):
    env = env_qtable()

    q_table = QLearningAgent.load_q_table_from_csv(file_name)
    agent = QLearningAgent(q_table=q_table, epsilon=0.0)  # epsilon=0.0 to ensure no exploration

    env.reset()
    while not env.is_done:
        if env.get_current_player() != 1:
            state = str(env.get_player_observations())
            possible_action = env.get_moves()
            move = agent.choose_action(state, possible_action)
        else:
            move = env.set_players(player_1_mode='human_gui')

        next_state, rewards, done, _, info = env.step(move)
        env.render('terminal_display')

        if done:
            print(env.get_game_status())
            break
        else:
            env.current_step += 1


def evaluate_agent(q_table_file, num_games, env):
    agent = QLearningAgent(q_table=QLearningAgent.load_q_table_from_csv(q_table_file), epsilon=0.0)  # Fully exploit
    wins = 0
    for _ in range(num_games):
        env.reset()
        done = False
        while not done:
            state = str(env.get_player_observations())
            possible_actions = env.get_moves()
            if env.get_current_player() == 1:
                action = agent.choose_action(state, possible_actions)
            else:
                action = env.get_action_random()
            _, rewards, done, _, info = env.step(action)
            if done and rewards['player_1'] > 0:
                wins += 1
    return wins / num_games


def collect_results(q_table_file, num_games, num_trials):
    env = ConnectGameEnv(connect=3, width=3, height=3, reward_winner=3, reward_loser=-3, living_reward=-0.1)
    results = []
    for _ in range(num_trials):
        win_rate = evaluate_agent(q_table_file, num_games, env)
        results.append(win_rate)
    return results

def plot_results(results):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=results, color='lightblue', fliersize=5, width=0.6)  # Use seaborn to draw the box plot
    plt.title('Win Rate Distribution over Multiple Evaluations')
    plt.ylabel('Win Rate')
    plt.xlabel('Trial')

    # Calculate and plot the average win rate across all trials
    overall_average = np.mean(results)
    plt.axhline(overall_average, color='red', linestyle='dashed', linewidth=2)
    plt.text(len(results)-1, overall_average, f'Average: {overall_average:.2f}', color='red', va='center', ha='right', backgroundcolor='w')

    plt.show()

if __name__ == "__main__":
    # train_agent_env(10000)
    # play_with_q_table()
    # num_games = 15000
    # win_rate = evaluate_agent('q_table_player_1.csv', num_games, env_qtable())
    # print(f'wins  {win_rate} / {num_games}')
    # Plot results

    num_games = 100  # Number of games to evaluate per trial
    num_trials = 5  # Number of times to evaluate
    results = collect_results('q_table_player_1.csv', num_games, num_trials)
    plot_results(results)
