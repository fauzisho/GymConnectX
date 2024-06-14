import random

import numpy as np

from qlearning.static_scenario.AgentProxQLearning import AgentProxQLearning


class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros((3, 3))

    def reset(self):
        self.board = np.zeros((3, 3))
        return self.board

    def available_actions(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def step(self, action):
        row, col = action
        self.board[row, col] = 1  # Assume player 1 is always the agent
        done = self.check_win() or not self.available_actions()  # Simplified check
        reward = 1 if self.check_win() else 0
        next_state = self.board
        return next_state, reward, done

    def check_win(self):
        # Simple win checking logic for tic-tac-toe
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3 or abs(sum(self.board[:, i])) == 3:
                return True
        if abs(sum(self.board.diagonal())) == 3 or abs(sum(np.fliplr(self.board).diagonal())) == 3:
            return True
        return False


class TicTacToeFeatureExtractor:
    def __init__(self):
        self.feature_size = 18  # 9 cells * 2 features per cell

    def extract_features(self, state, action):
        # state is a 3x3 numpy array with values 0 (empty), 1 (player), or -1 (opponent)
        # action is assumed to be a tuple (row, col)
        features = np.zeros(self.feature_size)
        flat_index = action[0] * 3 + action[1]
        features[flat_index * 2] = 1 if state[action[0], action[1]] == 1 else 0
        features[flat_index * 2 + 1] = 1 if state[action[0], action[1]] == -1 else 0
        return features


def train_agent():
    feature_extractor = TicTacToeFeatureExtractor()
    agent = AgentProxQLearning(feature_extractor=feature_extractor)
    env = TicTacToeEnv()

    for episode in range(10000):
        state = env.reset()
        done = False
        while not done:
            possible_actions = env.available_actions()
            action = agent.choose_action(state, possible_actions)
            next_state, reward, done = env.step(action)
            agent.update_q_value(state, action, reward, next_state, env.available_actions())
            state = next_state
    agent.save_weights_to_csv(f'agent_weights_episode_final.csv')


def play_game(weights_file):
    # Load the environment and agent
    feature_extractor = TicTacToeFeatureExtractor()
    agent = AgentProxQLearning(feature_extractor=feature_extractor)
    agent.load_weights_from_csv(weights_file)
    env = TicTacToeEnv()

    state = env.reset()
    done = False
    while not done:
        # Agent makes a move
        possible_actions = env.available_actions()
        action = agent.choose_action(state, possible_actions)
        state, _, done = env.step(action)
        print("Agent's move:", action)
        print(state)

        if done:
            print("Game over! Result:", "Agent wins" if env.check_win() else "Draw")
            break

        # Opponent's move
        opponent_action = random_opponent_action(state)
        if opponent_action:
            state[opponent_action[0], opponent_action[1]] = -1  # Assume -1 is the opponent
            print("Opponent's move:", opponent_action)
            print(state)
            if env.check_win():
                print("Game over! Result: Opponent wins")
                done = True
        else:
            print("No more moves available. Draw!")
            done = True


def random_opponent_action(board):
    available_actions = [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]
    return random.choice(available_actions) if available_actions else None


if __name__ == "__main__":
    # train_agent()
    play_game('agent_weights_episode_final.csv')
