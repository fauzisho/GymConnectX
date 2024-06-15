import numpy as np

def calculate_hot_spots(W, H, C):
    hot_spots = np.zeros((W, H))

    for x in range(W):
        for y in range(H):
            # Horizontal lines
            horizontal_lines = max(0, min(W - C + 1, x + 1, W - x))

            # Vertical lines
            vertical_lines = max(0, min(H - C + 1, y + 1, H - y))

            # Diagonal lines (\searrow)
            diagonal_down_lines = max(0, min(min(x + 1, y + 1), min(W - x, H - y), C))

            # Diagonal lines (\nearrow)
            diagonal_up_lines = max(0, min(min(x + 1, H - y), min(W - x, y + 1), C))

            # Total potential winning lines
            hot_spots[x, y] = horizontal_lines + vertical_lines + diagonal_down_lines + diagonal_up_lines

    return hot_spots

# Example usage for 9x9 board with Connect3
hot_spots_9x9_connect3 = calculate_hot_spots(3, 3, 3)
print("Hot spots for 9x9 board with Connect3:")
print(hot_spots_9x9_connect3)

# Example usage for 4x4 board with Connect3
hot_spots_4x4_connect3 = calculate_hot_spots(4, 4, 0)
print("Hot spots for 4x4 board with Connect3:")
print(hot_spots_4x4_connect3)

# Example usage for 5x5 board with Connect4
hot_spots_5x5_connect4 = calculate_hot_spots(5, 5, 0)
print("Hot spots for 5x5 board with Connect4:")
print(hot_spots_5x5_connect4)
