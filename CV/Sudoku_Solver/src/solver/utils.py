import os
import sys
sys.path.append(os.path.dirname(os.path.abspath('.')))
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

    def __getitem__(self, index):
        return self.board[index]

    def __setitem__(self, index, value):
        self.board[index] = value

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

    def validate_item(self,row: int, col: int,) -> bool:
        """
        Validate the value at a specific cell.
        Check if the value does not violate the Sudoku rules.
        """          
        temp_row = self.board[row] 
        temp_col = self.board[:,col]
        # temp_col = [self.board[i][col] for i in range(9)]

        temp_row = [num for num in temp_row if num != 0]
        temp_col = [num for num in temp_col if num != 0]
        
        if not len(temp_row)==len(set(temp_row)):
            return False           
        if not len(temp_col)==len(set(temp_col)):
            return False           
        
        temp_box=[]
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                itm = self.board[start_row+i][start_col+j]
                if itm!=0:
                    temp_box.append(itm)
        if not len(temp_box)==len(set(temp_box)):
            return False           

        return True
    def get_board(self) -> np.ndarray:
        """Get the current Sudoku board."""
        return self.board
    
    def set_board(self, board: np.ndarray):
        """Set the Sudoku board."""
        self.board = board


def visualize_board(board: np.ndarray, initial_board: np.ndarray = None, filename: str = None):
    """Visualize the Sudoku board as an image, coloring solved numbers differently."""
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
                color = 'grey' if initial_board is not None and initial_board[i, j] == 0 else 'black'
                ax.text(j + 0.5, 8.5 - i, str(board[i, j]), fontsize=18, ha='center', va='center', color=color)

    # Remove axis
    ax.axis('off')

    # Save or show the board
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    return fig

def is_unique(arr: np.ndarray) -> bool:
    """Check if all non-zero elements in the array are unique."""
    arr = arr[arr > 0]
    return len(arr) == len(set(arr))

if __name__ == "__main__":
    board = Board()

    board.set_board(np.array([[5,3,0,0,7,0,0,0,0],
                      [6,0,0,1,9,5,0,0,0],
                      [0,9,8,0,0,0,0,6,0],
                      [8,0,0,0,6,0,0,0,3],
                      [4,0,0,8,0,3,0,0,1],
                      [7,0,0,0,2,0,0,0,6],
                      [0,6,0,0,0,0,2,8,0],
                      [0,0,0,4,1,9,0,0,5],
                      [0,0,0,0,8,0,0,7,9]]))
    print(board.get_board())

    print(board.validate_item(0,0))
    visualize_board(board.get_board())

