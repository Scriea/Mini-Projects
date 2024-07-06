import numpy
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

from src.solver.utils import *
from src.solver.solver import *
from src.solver.generator import *


if __name__ =="__main__":
    ## Generate a random Sudoku board
    b = generate_random_board(difficulty="hard")                    # Set difficulty level to "easy", "medium", or "hard"
    fig = visualize_board(b)
    plt.show()

    ## Solve the Sudoku board
    solver = SudokuSolver()
    solver.set_board(b.copy())
    solver.solve()

    ## Visualize the solved board
    fig = visualize_board(board=solver.get_board(), initial_board=b)
    plt.show()
    
    

