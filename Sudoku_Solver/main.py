import numpy
import sys

from src.sudoku.utils import *
from src.sudoku.solver import *


if __name__ =="__main__":

    board = Board()
    board.generate_random_board(difficulty="medium")
    visualize_board(board.get_board())
    

