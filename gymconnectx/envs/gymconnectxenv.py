import os
import re
import urllib
import urllib.request
from typing import List
import io
import gym
from gym.spaces import Box, Discrete, Tuple
import imgBase64
import pygame
import sys
import numpy as np


class PyGameRenderEnv:
    def __init__(self, game_env, square_size, avatar_player_1=None, avatar_player_2=None):
        """
        Initialize the Connect Game GUI.

        Parameters:
            game_env: An instance of the Connect Game environment.
            square_size (int): The size of each square on the game board. Default is 100.
            avatar_player_1: avatar image player 1
            avatar_player_2: avatar image player 2
        """
        self.game_env = game_env
        self.square_size = square_size
        self.circle_radius = int(square_size / 2 - 5)

        pygame.init()
        self.width = self.game_env.width * square_size
        self.height = (self.game_env.height + 1) * square_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Connect Game")

        self.colors = {
            'black': (0, 0, 0),
            'silver': (192, 192, 192),
            'gold': (255, 215, 0),
            'brown': (130, 76, 28),
        }

        self.font = pygame.font.SysFont("monospace", 25)
        self.preview_index = -1

        self.avatar_player_1 = self.load_avatar(avatar_player_1)
        self.avatar_player_2 = self.load_avatar(avatar_player_2)

    def is_url(self, string):
        """Check if the string is a URL."""
        return string.startswith(('http://', 'https://'))

    def is_base64(self, string):
        """Check if the string is in base64 format."""
        if string.startswith('data:image/'):
            return True
        # Additional checks can be added to validate base64 using regex
        # This regex checks if it contains only base64 characters and ends with '='
        return bool(re.match('^[A-Za-z0-9+/]+={0,2}$', string.split(',')[-1]))

    def is_path(self, string):
        """Check if the string is a file path. This is usually the fallback option."""
        return os.path.isfile(string)

    def load_image_from_path(self, path):
        """Loads an image from a local file path and resizes it."""
        return self.process_image(pygame.image.load(path))

    def load_avatar(self, avatar_string):
        """Determines the type of the avatar input and loads the image accordingly."""
        if avatar_string is None:
            return None
        elif self.is_base64(avatar_string):
            return self.load_image_from_base64(avatar_string)
        elif avatar_string is not None:
            return self.load_image_from_path(avatar_string)
        return None

    def load_image_from_base64(self, base64_string):
        """Helper method to convert a base64 string to a pygame image and resize it to fit within the game piece circle."""
        image_data = base64.b64decode(base64_string)
        image_io = io.BytesIO(image_data)
        image = pygame.image.load(image_io)
        return self.process_image(image)

    def process_image(self, image):
        """Helper method to resize an image to fit within the game piece circle."""
        circle_diameter = self.circle_radius * 2
        new_width = int(circle_diameter * 0.7)
        new_height = int(circle_diameter * 0.7)
        resized_image = pygame.transform.scale(image, (new_width, new_height))
        return resized_image

    def draw_board(self):
        """
        Draw the game board on the screen.

        This method iterates through each position on the game board and draws the squares and pieces accordingly.
        """
        for c in range(self.game_env.width):
            for r in range(self.game_env.height):
                pygame.draw.rect(self.screen, self.colors['brown'],
                                 (c * self.square_size, (self.game_env.height - r) * self.square_size,
                                  self.square_size, self.square_size))

                piece = self.game_env.board[c][r]
                color = self.colors['black'] if piece == -1 else self.colors['silver'] if piece == 0 else self.colors[
                    'gold']

                # Center coordinates for the circle
                center_x = int(c * self.square_size + self.square_size / 2)
                center_y = int((self.game_env.height - r) * self.square_size + self.square_size / 2)

                # Draw the circle for the game piece
                pygame.draw.circle(self.screen, color, (center_x, center_y), self.circle_radius)
                self.show_avatar(piece, color, center_x, center_y)

        pygame.display.update()

    def show_avatar(self, piece, color, center_x, center_y):
        """
            Displays the avatar for a player at a specified location on the game board.

            This method checks the game piece value and determines which player's avatar
            to display. If the piece matches a player (0 for Player 1, 1 for Player 2),
            it displays the corresponding avatar centered at the specified coordinates.

            Parameters:
                piece (int): The value of the game piece at the current board location,
                             which indicates which player's piece, if any, is at this location.
                             Typically, 0 might represent Player 1 and 1 might represent Player 2.
                color (tuple): The color tuple (R, G, B) representing the color currently
                               used for drawing purposes. This parameter is logged but not used
                               for drawing in this method.
                center_x (int): The x-coordinate of the center where the avatar should be placed.
                center_y (int): The y-coordinate of the center where the avatar should be placed.
        """
        # Determine the correct avatar based on the piece
        avatar = None
        if piece == 0:
            avatar = self.avatar_player_1
        elif piece == 1:
            avatar = self.avatar_player_2

        # If a valid avatar is found, draw it centered at the specified location
        if avatar:
            avatar_rect = avatar.get_rect()
            avatar_rect.center = (center_x, center_y)
            self.screen.blit(avatar, avatar_rect)

    def update_display(self):
        """
        Update the display to reflect the current game state.

        This method clears the screen, fills it with the background color, and then redraws the game board with the
            current state of the game.

        """
        self.screen.fill(self.colors['black'])
        self.draw_board()
        pygame.display.update()

    def update_display_click(self, preview_col=None):
        """
        Update the display after a mouse click event.

        Parameters:
            preview_col (int): The column index of the preview move. Default is None.
        """
        self.draw_board()
        if preview_col is not None:
            if self.preview_index == -1:
                self.preview_index = preview_col
            else:
                circle_x_position = self.preview_index * self.square_size + self.square_size // 2
                circle_y_position = self.circle_radius
                pygame.draw.circle(self.screen, self.colors['black'], (circle_x_position, circle_y_position),
                                   self.circle_radius)
                self.preview_index = preview_col

            preview_color = self.colors['gold'] if self.game_env.current_player == 1 else self.colors['silver']
            circle_x_position = preview_col * self.square_size + self.square_size // 2
            circle_y_position = self.circle_radius
            pygame.draw.circle(self.screen, preview_color, (circle_x_position, circle_y_position), self.circle_radius)
            self.show_avatar(self.game_env.current_player, preview_color, circle_x_position, circle_y_position)
        pygame.display.update()

    def handle_events(self):
        """
        Handle Pygame events.

        This method processes events from the Pygame event queue. If a quit event is detected, it stops the game.

        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.stop()

    def handle_events_click(self):
        """
        Handle Pygame events for mouse clicks.

        This method continuously processes events from the Pygame event queue. If a quit event is detected, it stops the game.
        If a mouse motion event occurs, it updates the display to show a preview of the move. If a mouse button down event
        occurs, it calculates the column index based on the mouse position and checks if it's a valid move. If it's valid,
        it returns the column index. Otherwise, it shows an alert message prompting the user to select another column.

        Returns:
            int: The column index of the selected move.
        """
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.stop()
                elif event.type == pygame.MOUSEMOTION:
                    xpos = event.pos[0]
                    preview_col = xpos // self.square_size
                    self.update_display_click(preview_col=preview_col)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    xpos = event.pos[0]
                    col = xpos // self.square_size
                    if col in self.game_env.get_moves():
                        return col
                    else:
                        self.show_alert("Please select another column.")

    def show_alert(self, message):
        """
        Display an alert message on the screen.

        Parameters:
            message (str): The message to be displayed in the alert.

        """
        text = self.font.render(message, True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.width // 20, self.height // 2))
        box_width = text_rect.width + 20
        box_height = text_rect.height + 20
        box_surface = pygame.Surface((box_width, box_height))
        box_surface.fill((255, 0, 0))
        box_rect = box_surface.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(box_surface, box_rect)
        text_rect.center = box_rect.center
        self.screen.blit(text, text_rect)
        pygame.display.update()
        pygame.time.delay(2000)

    def stop(self):
        """
        Stop the game and exit Pygame.

        This method quits the Pygame module and exits the program.

        """
        pygame.quit()
        sys.exit()


class ConnectGameEnv(gym.Env):
    def __init__(self, connect=4, width=7, height=7, reward_winner=1, reward_loser=-1, living_reward=0, max_steps=100,
                 delay=100, square_size=100, avatar_player_1=None, avatar_player_2=None):
        """
        Initializes a new ConnectGameEnv, which is a gaming environment for playing games like Connect Four.

        :param connect: Number of consecutive tokens required to win (default is 4).
        :param width: Width of the game board (number of columns, default is 7).
        :param height: Height of the game board (number of rows, default is 7).
        :param reward_winner: Reward given to the winner at the end of the game (default is 1).
        :param reward_loser: Reward (penalty) given to the loser at the end of the game (default is -1).
        :param living_reward: Reward given at each step of the game, applicable to all ongoing games (default is 0).
        :param max_steps: Maximum number of steps the game can take before ending (default is 100).
        :param delay: Time delay (in milliseconds) between moves, primarily used for GUI purposes (default is 100).
        :param avatar_player_1: avatar image player 1
        :param avatar_player_2: avatar image player 2
        Initializes the environment with the specified dimensions and settings. It sets up spaces for observations
        and actions based on the game rules, as well as initializing a renderer for graphical display.
        """
        self.connect = connect
        self.width = width
        self.height = height

        self.reward_loser = reward_loser
        self.reward_winner = reward_winner
        self.living_reward = living_reward

        self.max_steps = max_steps
        self.current_step = 0
        self.is_done = False

        self.delay = delay

        self.observation_space = Box(low=0, high=255, shape=(height, width, 1), dtype=np.uint8)
        self.action_space = Tuple([Discrete(self.width) for _ in range(2)])
        self.state_space_size = (self.height * self.width) ** 3

        self.renderer = PyGameRenderEnv(self, square_size, avatar_player_1, avatar_player_2)
        self.reset()

    def step(self, movecol):
        """
        Processes a move made by the current player by placing a chip in the specified column, then checks for game termination.

        :param movecol: The column number (zero-indexed) where the current player wishes to place their chip.

        :return: A tuple containing four elements:
            - Observations: Current state of the board from the perspective of both players.
            - Reward_players: A dictionary detailing the rewards for player 1 and player 2 based on the latest move.
            - Is_done: Boolean value indicating whether the game has ended (either by a win or a full board).
            - Info: A dictionary containing additional information such as legal actions for the next move and the next player.

        Raises:
            IndexError: If the move is invalid (e.g., the column is full or out of bounds).

        This method updates the game state by inserting a chip into the chosen column, checks for a winner, updates the current
        player, and calculates the rewards based on the state of the game. It also updates the display through the renderer and
        prepares the info dictionary for the next player's move.
        """
        if not (0 <= movecol < self.width and self.board[movecol][self.height - 1] == -1):
            raise IndexError(
                f'Invalid move. tried to place a chip on column {movecol} which is already full. Valid moves are: {self.get_moves()}')

        row = self.height - 1
        while row >= 0 and self.board[movecol][row] == -1:
            row -= 1
        row += 1

        self.board[movecol][row] = self.current_player
        self.current_player = 1 - self.current_player

        self.winner, reward_vector = self.check_for_episode_termination(movecol, row)

        info = {'legal_actions': self.get_moves(), 'next_player': self.current_player + 1}
        reward_players = {'player_1': reward_vector[0], 'player_2': reward_vector[1]}
        self.is_done = self.winner is not None
        self.renderer.update_display()

        obs = self.get_player_observations()
        terminated = self.is_done
        truncated = False

        return obs, reward_players, terminated, truncated, info

    def reset(self) -> List[np.ndarray]:
        """
        Resets the game environment to its initial state. This method is typically called at the start of each new game.

        :return: A list of numpy arrays representing the initial observations of the game board for both players.
                     Each observation is an ndarray showing the current state of the board from the player's perspective.

        This method initializes the game board to an empty state, sets the current player to player 1 (represented as 0),
        and clears any previous game winner. It also resets the current step count and game completion flag. The display
        is updated to reflect the reset state through the renderer.
        """
        self.board = np.full((self.width, self.height), -1)
        self.current_player = 0
        self.winner = None
        self.current_step = 0
        self.is_done = False
        self.renderer.update_display()
        return self.get_player_observations()

    def render(self, mode='terminal_display'):
        """
        Renders the game board.

        Parameters:
            mode (str): The rendering mode. It can be one of the following:
                - 'terminal_display': Renders the game board in the terminal.
                - 'gui_handle_event': Handles events for GUI display.
                - 'gui_update_display': Updates the GUI display.
                - 'gui_handle_event_click': Handles mouse click events for GUI display.

        Raises:
            NotImplementedError: If the rendering mode is not supported.
        """
        if mode == 'terminal_display':
            s = ""
            for x in range(self.height - 1, -1, -1):
                for y in range(self.width):
                    s += {-1: '.', 0: 'X', 1: 'O'}[self.board[y][x]]
                s += "\n"
            print(s)
        elif mode == 'gui_handle_event':
            self.renderer.handle_events()
        elif mode == 'gui_update_display':
            self.renderer.update_display()
            pygame.time.wait(self.delay)
        elif mode == 'gui_handle_event_click':
            self.renderer.handle_events_click()
        else:
            raise NotImplementedError('Rendering mode not supported')

    def close(self):
        """
        Closes the game window.

        This method stops the renderer, effectively closing the game window.

        """
        self.renderer.stop()

    def get_player_observations(self) -> np.ndarray:
        """
           Get the observations of the current game state from the perspective of the players.

           Returns:
               np.ndarray: A 2D array representing the observations of the game state. Each element in the array
               represents the state of a position on the game board from the perspective of the players. The array
               has a shape of (width, height), where width and height are the dimensions of the game board.

        """
        observation = np.empty((self.width, self.height), dtype='<U1')
        for x in range(self.height - 1, -1, -1):
            for y in range(self.width):
                observation[y][x] = {-1: '.', 0: 'X', 1: 'O'}[self.board[y][x]]
        return observation

    def check_for_episode_termination(self, movecol, row):
        """
        Checks if the current game episode has terminated.

        Parameters:
            movecol (int): The column index of the last move made.
            row (int): The row index of the last move made.

        Returns:
            Tuple[int, List[int]]: A tuple containing the winner of the game episode and the reward vector.
                - The winner (int) can be one of the following:
                    - 0: Player 1 wins.
                    - 1: Player 2 wins.
                    - -1: The game is a draw.
                - The reward vector (List[int]) contains the rewards for both players. The first element is
                  the reward for Player 1, and the second element is the reward for Player 2.
        """
        winner, reward_vector = self.winner, [self.living_reward, self.living_reward]
        if self.does_move_win(movecol, row):
            winner = 1 - self.current_player
            if winner == 0:
                reward_vector = [self.reward_winner, self.reward_loser]
            elif winner == 1:
                reward_vector = [self.reward_loser, self.reward_winner]
        elif self.get_moves() == []:
            winner = -1
            reward_vector = [0, 0]
        return winner, reward_vector

    def get_moves(self):
        """
           Get the available moves for the current game state.

           Returns:
               List[int]: A list containing the column indices of the available moves.
        """
        if self.winner is not None:
            return []
        return [col for col in range(self.width) if self.board[col][self.height - 1] == -1]

    def does_move_win(self, x, y):
        """
            Checks if the last move results in a win.

            Parameters:
                x (int): The column index of the last move.
                y (int): The row index of the last move.

            Returns:
                bool: True if the last move results in a win, False otherwise.
        """
        me = self.board[x][y]
        for (dx, dy) in [(0, +1), (+1, +1), (+1, 0), (+1, -1)]:
            p = 1
            while self.is_on_board(x + p * dx, y + p * dy) and self.board[x + p * dx][y + p * dy] == me:
                p += 1
            n = 1
            while self.is_on_board(x - n * dx, y - n * dy) and self.board[x - n * dx][y - n * dy] == me:
                n += 1

            if p + n >= (self.connect + 1):
                return True

        return False

    def is_on_board(self, x, y):
        """
        Checks if a given position is within the bounds of the game board.

        Parameters:
            x (int): The column index.
            y (int): The row index.

        Returns:
            bool: True if the position is within the bounds of the game board, False otherwise.
        """
        return x >= 0 and x < self.width and y >= 0 and y < self.height

    def get_result(self, player):
        """
        Get the result of the game for a specific player.

        Parameters:
            player (int): The player for whom the result is being queried.

        Returns:
            int: The result of the game for the specified player. It returns:
                - 0 if the game is a draw.
                - +1 if the specified player is the winner.
                - -1 if the specified player is the loser.
        """
        if self.winner == -1: return 0
        return +1 if player == self.winner else -1

    def set_players(self, player_1_mode='random', player_2_mode='random'):
        """
        Set the modes for Player 1 and Player 2.

        Parameters:
            player_1_mode (str): The mode for Player 1. Default is 'random'.
            player_2_mode (str): The mode for Player 2. Default is 'random'.

        Returns:
            int: The move made by the current player based on their mode.
        """
        print(f"Player: {self.get_current_player()}, ")
        if self.current_player == 0:
            move = self.set_switch_player(player_1_mode)
        else:
            move = self.set_switch_player(player_2_mode)

        return move

    def set_switch_player(self, mode):
        """
        Set the action for the current player based on the specified mode.

        Parameters:
            mode (str): The mode indicating how the current player's action should be determined. It can be one of the following:
                - 'random': Player action is determined randomly.
                - 'human_terminal': Player action is determined through the terminal.
                - 'human_gui': Player action is determined through a graphical user interface.

        Returns:
            int or None: The move made by the current player based on their mode. If the mode is not recognized,
            None is returned.
        """
        if mode == 'human_terminal':
            return self.get_action_human_terminal()
        elif mode == 'random':
            return self.get_action_random()
        elif mode == 'human_gui':
            return self.get_action_human_gui()
        else:
            return None

    def get_action_human_gui(self):
        """
        Get the action chosen by a human player through a graphical user interface.

        Returns:
            int or None: The column index representing the chosen move. Returns None if no valid move is selected.
        """
        col = None
        while col is None:
            self.render(mode='gui_handle_event_click')
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                col = pos[0] // self.renderer.square_size
                if col in self.get_moves():
                    return col
        return col

    def get_action_random(self):
        """
        Get a random action chosen by the computer player.

        Returns:
            int or None: The column index representing the randomly chosen move. Returns None if there are no valid moves.
        """
        self.render(mode='gui_handle_event')
        if not self.get_moves():
            print("No more valid moves. It's a draw or the game is won.")
            return None
        else:
            return np.random.choice(self.get_moves())

    def get_action_human_terminal(self):
        """
        Get the action chosen by a human player through the terminal.

        Returns:
            int: The column index representing the chosen move.
        """
        self.render(mode='gui_handle_event')
        print("Available moves:", self.get_moves())
        step_move = None
        while step_move not in self.get_moves():
            try:
                step_move = int(input("Enter your move (odd column number): "))
                if step_move not in self.get_moves():
                    self.renderer.show_alert("Please select another column.")
                    print("That column is full or invalid. Try again.")
            except ValueError:
                print("Please enter a valid integer.")

        return step_move

    def get_current_player(self):
        """
        Get the index of the current player.
        Player 1 return 1
        Player 2 return 2

        Returns:
            int: The index of the current player. Player indices start from 1.
        """
        return self.current_player + 1

    def get_game_status(self):
        """
        Get the status of the game.

        Returns:
            str: A message indicating the current status of the game. It can be one of the following:
                - "The game is ongoing. Player {current_player + 1} Move": If the game is ongoing.
                - "The game is a draw.": If the game is a draw.
                - "Player {winner + 1} wins!": If a player has won the game.
        """
        if self.winner is None:
            return f"The game is ongoing. Player {self.current_player + 1} Move"
        elif self.winner == -1:
            return "The game is a draw."
        else:
            return f"Player {self.winner + 1} wins!"
