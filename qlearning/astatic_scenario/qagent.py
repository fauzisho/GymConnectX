import random
import csv
from gymconnectx.envs import ConnectGameEnv


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}  # Q-table initially empty
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


def train_agent_env():
    env = ConnectGameEnv(
        connect=3,
        width=3,
        height=3,
        reward_winner=3,
        reward_loser=-3,
        reward_living=-0.1,)

    agent = QLearningAgent()

    for game in range(100):
        env.reset()
        scenario_player_1_and_2 = [1, 0, 1, 0]
        state = ""
        action = -1
        while not env.is_done:
            try:
                if env.current_step == 4:
                    state = str(env.get_player_observations())
                    possible_action = env.get_moves()
                    move = agent.choose_action(state, possible_action)
                    action = move
                else:
                    if env.current_step >= 5:
                        if 0 in env.get_moves():
                            move = 0
                        elif 1 in env.get_moves():
                            move = 1
                        else:
                            move = 2
                    elif env.get_current_player() == 1:
                        move = scenario_player_1_and_2[env.current_step]
                    else:
                        move = scenario_player_1_and_2[env.current_step]

                next_state, rewards, done, _, info = env.step(move)

                if done or env.get_current_player() == 2 and action != -1:
                    env.render(mode='terminal_display')
                    reward_player_1 = rewards['player_1']
                    print(env.get_game_status())
                    print(f'game: {game}, action: {action}, reward: {reward_player_1}')
                    agent.update_q_value(state, action, reward_player_1, next_state)

                if done:
                    print("------------------------")
                    break
                else:
                    env.current_step += 1

            except Exception as e:
                print(f"An error occurred: {str(e)}")
                break

    # Save the Q-table to CSV after training
    agent.save_q_table_to_csv()


if __name__ == "__main__":
    train_agent_env()
