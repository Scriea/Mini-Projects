import os
import sys
sys.path.append(os.path.dirname(os.path.abspath('.')))


import numpy as np
from .utils import Board,visualize_board
import ctypes
from typing import List
import matplotlib.pyplot as plt


class SudokuSolver():
    def __init__(self,board=None):
        if board is None:
            self.board = Board()
        else:
            self.board = board

    def set_board(self, board: np.ndarray):
        """Set the Sudoku board."""
        self.board.set_board(board)

    def get_board(self) -> np.ndarray:
        """Get the current Sudoku board."""
        return self.board.get_board()
    
    def is_valid(self) -> bool:
        """Check if the current board is valid."""
        return self.board.validate()    

    def solve(self):
        """
        Solve the Sudoku board.
        Arguments:
            board: np.ndarray
                The Sudoku board to solve. If None, the current board will be used.
        """       
        self._solve()


    """
    Brute force algorithm to solve the Sudoku board.
    """
    def _solve(self):
        for row in range(9):
            for col in range(9):
                if self.board[row][col]==0:
                    for c in range(1,10):
                        self.board[row][col]=str(c)
                        if self.board.validate_item(row, col):
                            if self._solve():
                                return True
                        self.board[row][col]=0
                    return False
        return True       




if __name__=="__main__":
    board=Board()
    b = np.array([[5,3,0,0,0,0,0,0,0],
                  [6,0,0,1,9,5,0,0,0],
                  [0,9,8,0,0,0,0,6,0],
                  [8,0,0,0,6,0,0,0,3],
                  [4,0,0,8,0,3,0,0,1],
                  [7,0,0,0,2,0,0,0,6],
                  [0,6,0,0,0,0,2,8,0],
                  [0,0,0,4,1,9,0,0,5],
                  [0,0,0,0,8,0,0,7,9]])
    print(b)
    board.set_board(b.copy())
       
    solver = SudokuSolver(board)
    solver.solve()

    print(solver.is_valid())

    print(solver.get_board())
    fig = visualize_board(b)
    plt.show()
    fig = visualize_board(board=solver.get_board(), initial_board=b)
    plt.show()