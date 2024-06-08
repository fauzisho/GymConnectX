import random
import csv
import os


class AgentQLearning:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0, role='player_1', ):
        self.q_table = {}  # Q-table initially empty
        self.role = role  # Player role: 'player_1' or 'player_2'
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def get_state_key(self, state):
        """Convert state to a string key for Q-table."""
        return str(state)

    def choose_action(self, state, possible_actions) -> int:
        """Choose an action using an epsilon-greedy policy."""
        state_key = self.get_state_key(state)
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        else:
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

        next_max = max(self.q_table[next_state_key].values()) if next_state_key in self.q_table and self.q_table[
            next_state_key] else 0
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

    def test_agent_with_q_table(self, scenario_player_1_and_2, env, file_name):
        q_table = self.load_q_table_from_csv(file_name)
        self.q_table = q_table

        env.reset()
        while not env.is_done:
            if env.current_step < 4:
                action = scenario_player_1_and_2[env.current_step]  # based on scenario
            else:
                if env.get_current_player() != 1:
                    state = str(env.get_player_observations())
                    possible_action = env.get_moves()
                    action = self.choose_action(state, possible_action)
                else:
                    # action = env.set_players(player_2_mode='human_gui')
                    action = env.set_players(player_1_mode='human_gui')

            next_state, rewards, done, _, info = env.step(action)
            env.render('terminal_display')

            if done:
                print(env.get_game_status())
                break
            else:
                env.current_step += 1
