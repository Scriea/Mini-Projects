import os
import sys
sys.path.append(os.path.dirname(os.path.abspath('.')))

import random
import numpy as np
import matplotlib.pyplot as plt 

from .utils import *
from .solver import *


def generate_random_board(difficulty: str = "easy") -> np.ndarray:
        """
        Generate a random Sudoku board with a given difficulty.
        Solves a Sudoku board and removes numbers to create a puzzle.

        See sources:
            https://dingo.sbs.arizona.edu/~sandiway/sudoku/examples.html
            https://dlbeer.co.nz/articles/sudoku.html
        """
        board = np.zeros((9, 9), dtype=int)
        
        for i in range(0, 9, 3):
            nums = list(range(1, 10))
            random.shuffle(nums)
            for row in range(3):
                for col in range(3):
                    board[i + row][i + col] = nums.pop()

        solver = SudokuSolver()
        solver.set_board(board)
        solver.solve()
        if difficulty == "easy":
            N=35
        elif difficulty == "medium":
            N=45
        elif difficulty == "hard":
            N=60
        indices_to_remove = generate_random_tuples(N)

        for index in indices_to_remove:
            i,j = index
            board[i][j] = 0

        return board

def generate_random_tuples(n, grid_size=9):
    """Generate n unique random tuples (i, j) from a grid of size grid_size x grid_size."""
    if n > grid_size * grid_size:
        raise ValueError("Number of tuples requested exceeds the total number of cells in the grid.")
    
    tuples = set()
    while len(tuples) < n:
        i = random.randint(0, grid_size - 1)
        j = random.randint(0, grid_size - 1)
        tuples.add((i, j))
    
    return list(tuples)
