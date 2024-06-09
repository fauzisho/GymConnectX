import os
import random
import csv
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from gymconnectx.envs import ConnectGameEnv


class QLearningAgent:
    def __init__(self, role='player_1', q_table={}, alpha=0.1, gamma=0.9):
        self.role = role  # Player role: 'player_1' or 'player_2'
        self.q_table = q_table  # Q-table initially empty
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor

    def get_state_key(self, state):
        """Convert state to a string key for Q-table."""
        return str(state)

    def choose_action(self, state, possible_actions, epsilon=0.0) -> int:
        """Choose an action using an epsilon-greedy policy."""
        state_key = self.get_state_key(state)
        if random.random() < epsilon:  # Exploration
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


def play_with_q_table(file_name="q_table_player_2.csv"):
    env = ConnectEnv()

    q_table = QLearningAgent.load_q_table_from_csv(file_name)
    agent = QLearningAgent(q_table=q_table)

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


def ConnectEnv():
    return ConnectGameEnv(connect=3, width=3, height=3, reward_winner=1, reward_loser=-1, reward_living=0)


def train_agent_env(num_training, num_trials, num_exploit, games_per_segment):
    total_games = num_training + num_exploit
    num_segments = total_games // games_per_segment
    segment_results = {i: [] for i in range(num_segments)}
    segment = 0

    env = ConnectEnv()
    roles = ['player_1', 'player_2']
    agent_1 = QLearningAgent(roles[0])
    agent_2 = QLearningAgent(roles[1])
    last_state = {role: None for role in roles}
    last_action = {role: None for role in roles}

    for game in range(total_games):
        if (game + 1) > num_training:
            epsilon = 0.0
        else:
            epsilon = 0.1
        env.reset()
        done = False
        # start training game
        while not done:
            current_role = roles[env.get_current_player() - 1]
            if env.get_current_player() == 1:
                agent = agent_1
            else:
                agent = agent_2

            state = str(env.get_player_observations())
            possible_actions = env.get_moves()
            action = agent.choose_action(state, possible_actions, epsilon)

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

                if (game + 1) % games_per_segment == 0:
                    q_table_name_agent_1 = f'q_table_player_1_{game + 1}.csv'
                    q_table_name_agent_2 = f'q_table_player_2_{game + 1}.csv'
                    agent_1.save_q_table_to_csv(q_table_name_agent_1)
                    agent_2.save_q_table_to_csv(q_table_name_agent_2)

                    for _ in range(num_trials):
                        win_rate = evaluate_agent(q_table_name_agent_1, 100, env, epsilon)
                        segment_results[segment].append(win_rate)
                    segment = segment + 1

    plot_results(segment_results)


def evaluate_agent(q_table_file, num_games, env, epsilon):
    agent = QLearningAgent(q_table=QLearningAgent.load_q_table_from_csv(q_table_file))  # Fully exploit
    wins = 0
    for _ in range(num_games):
        env.reset()
        done = False
        while not done:
            state = str(env.get_player_observations())
            possible_actions = env.get_moves()
            if env.get_current_player() == 1:
                action = agent.choose_action(state, possible_actions, epsilon)
            else:
                action = env.get_action_random()
            _, rewards, done, _, _ = env.step(action)
            if done and rewards['player_1'] > 0:
                wins += 1
    return wins / num_games


def plot_results(segmented_results):
    plt.figure(figsize=(10, 6))
    data = []
    labels = []
    averages = []

    # Prepare data for the box plot
    for i, results in segmented_results.items():
        data.append(results)
        labels.append(f'{(i + 1) / 2}')
        averages.append(np.mean(results))

    # Plot boxplot
    sns.boxplot(data=data)
    plt.title('Win Rate Across Different Game Segments')
    plt.ylabel('Q Player Win Rate %')
    plt.xticks(range(len(labels)), labels)  # Set custom x-axis labels

    # Plotting the average line with markers
    plt.plot(averages, 'r--o', label='Average Win Rate')  # Red dashed line with circle markers
    plt.legend()

    # Annotating each average point
    for i, avg in enumerate(averages):
        plt.text(i, avg, f'{avg * 100:.2f}%', color='red', ha='center')

    plt.ylim(0, 1)  # Adjust the Y-axis limits to be between 0 and 1 (0% to 100%)
    plt.show()


def epsilon_value(segment, num_segment_explore):
    if segment > num_segment_explore:
        epsilon = 0.0
    else:
        epsilon = 0.1
    return epsilon


def collect_results(q_table_files, num_training, num_exploit, num_trials, games_per_segment):
    total_games = (num_training + num_exploit)
    num_segment_explore = num_training // games_per_segment
    env = ConnectEnv()
    num_segments = total_games // games_per_segment
    segment_results = {i: [] for i in range(num_segments)}

    for _ in range(num_trials):
        for segment in range(num_segments):
            start_game = segment * games_per_segment + 1
            end_game = min((segment + 1) * games_per_segment, total_games)
            q_table_index = start_game // games_per_segment
            q_table_file = q_table_files[q_table_index - 1]  # Adjust index since list index starts from 0
            win_rate = evaluate_agent(q_table_file, end_game - start_game + 1, env,
                                      epsilon_value(segment, num_segment_explore))
            segment_results[segment].append(win_rate)

    return segment_results


if __name__ == "__main__":
    play_with_q_table(file_name="q_table_player_2.csv")

    num_training = 30000  # Total number of games to evaluate
    num_exploit = 15000
    games_per_segment = 3000  # Number of games per segment to show in the box plot
    num_trials = 5  # Number of times to evaluate

    # train_agent_env(num_training=num_training,
    #                 num_exploit=num_exploit,
    #                 num_trials=num_trials,
    #                 games_per_segment=games_per_segment)

    # q_table_files = [
    #     'q_table_player_1_3000.csv',
    #     'q_table_player_1_6000.csv',
    #     'q_table_player_1_9000.csv',
    #     'q_table_player_1_12000.csv',
    #     'q_table_player_1_15000.csv',
    #     'q_table_player_1_18000.csv',
    #     'q_table_player_1_21000.csv',
    #     'q_table_player_1_24000.csv',
    #     'q_table_player_1_27000.csv',
    #     'q_table_player_1_30000.csv',
    #     'q_table_player_1_33000.csv',
    #     'q_table_player_1_36000.csv',
    #     'q_table_player_1_39000.csv',
    #     'q_table_player_1_42000.csv',
    #     'q_table_player_1_45000.csv'
    # ]
    #
    # segmented_results = collect_results(q_table_files, num_training, num_exploit, num_trials, games_per_segment)
    # plot_results(segmented_results)
