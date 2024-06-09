import random
import csv
import os


class AgentQLearning:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0, role='player_1', q_table=None):
        self.q_tables = {'player_1': {}, 'player_2': {}}  # Separate Q-tables for each player role
        self.role = role  # Player role: 'player_1' or 'player_2'
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        if q_table is not None:
            self.q_tables[self.role] = q_table

    def get_state_key(self, state):
        return str(state)

    def choose_action(self, state, possible_actions):
        state_key = self.get_state_key(state)
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        else:
            q_table = self.q_tables[self.role]
            if state_key in q_table and q_table[state_key]:
                possible_q_values = {action: q_table[state_key][action] for action in possible_actions if
                                     action in q_table[state_key]}
                if possible_q_values:
                    return max(possible_q_values, key=possible_q_values.get)
                else:
                    return random.choice(possible_actions)
            else:
                return random.choice(possible_actions)

    def update_q_value(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        q_table = self.q_tables[self.role]

        if state_key not in q_table:
            q_table[state_key] = {}
        if action not in q_table[state_key]:
            q_table[state_key][action] = 0

        next_max = max(q_table[next_state_key].values()) if next_state_key in q_table and q_table[next_state_key] else 0
        q_table[state_key][action] += self.alpha * (reward + self.gamma * next_max - q_table[state_key][action])

    def save_q_table_to_csv(self, file_name):
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["state", "action", "q_value"])
            for state, actions in self.q_tables[self.role].items():
                for action, q_value in actions.items():
                    writer.writerow([state, action, q_value])

    @staticmethod
    def load_q_table_from_csv(file_name):
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
