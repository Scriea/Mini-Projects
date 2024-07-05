import random
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

class Board:
    def __init__(self, board=None):
        if board is None:
            self.board = np.zeros((9, 9), dtype=int)
        else:
            self.board = board

    def validate(self) -> bool:
        """
        Validate the Sudoku board.
        Check if all rows, columns, and 3x3 subgrids contain unique numbers.
        """
        for i in range(9):
            row = self.board[i, :]
            col = self.board[:, i]
            if not (is_unique(row) and is_unique(col)):
                return False
        for i in range(3):
            for j in range(3):
                subgrid = self.board[i*3:(i+1)*3, j*3:(j+1)*3].flatten()
                if not is_unique(subgrid):
                    return False
        return True


    ## Add difficulty levels
    def generate_random_board(self, difficulty: str = "easy") -> np.ndarray:
        """
        Generate a random Sudoku board with a given difficulty.
        """
        board = np.zeros((9, 9), dtype=int)
        
        for i in range(0, 9, 3):
            nums = list(range(1, 10))
            random.shuffle(nums)
            for row in range(3):
                for col in range(3):
                    board[i + row][i + col] = nums.pop()
        
        self.board = board
        return self.board

    def get_board(self) -> np.ndarray:
        """Get the current Sudoku board."""
        return self.board
    
    def set_board(self, board: np.ndarray):
        """Set the Sudoku board."""
        self.board = board


## Utility Functions
def visualize_board(board: np.ndarray, filename: str = None):
    """Visualize the Sudoku board as an image."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Draw the grid
    for i in range(10):
        if i % 3 == 0:
            ax.plot([0, 9], [i, i], color='black', linewidth=2)
            ax.plot([i, i], [0, 9], color='black', linewidth=2)
        else:
            ax.plot([0, 9], [i, i], color='black', linewidth=1)
            ax.plot([i, i], [0, 9], color='black', linewidth=1)

    # Fill the numbers
    for i in range(9):
        for j in range(9):
            if board[i, j] != 0:
                ax.text(j + 0.5, 8.5 - i, str(board[i, j]), fontsize=18, ha='center', va='center')

    # Remove axis
    ax.axis('off')

    # Save or show the board
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()

def is_unique(arr: np.ndarray) -> bool:
    """Check if all non-zero elements in the array are unique."""
    arr = arr[arr > 0]
    return len(arr) == len(set(arr))

if __name__ == "__main__":
    board = Board()
    board.generate_random_board("easy")
    print(board.get_board())
    visualize_board(board.get_board())
