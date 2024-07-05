import numpy as np
from .utils import Board


class Sudoku():
    def __init__(self, board=None):
        if board is None:
            self.board = Board()
        else:
            self.board = board



    def set_board(self, board: np.ndarray):
        """Set the Sudoku board."""
        self.board.set_board(board)

## Utils

