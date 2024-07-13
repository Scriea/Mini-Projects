import os
import sys
sys.path.append(os.path.dirname(os.path.abspath('.')))
import cv2
import matplotlib.pyplot as plt
import argparse
import numpy as np


from src.solver.utils import *
from src.solver.solver import *
from src.solver.generator import *
from src.cv import utils

if __name__ =="__main__":

    parser = argparse.ArgumentParser(description='Sudoku Solver')
    parser.add_argument("-d",   '--difficulty', type=str, default="easy", help='Difficulty level of the Sudoku board')
    parser.add_argument("-i",'--image', type=str, help='Path to the image of the Sudoku board', default=None)
    
    args = parser.parse_args()


    ## Generate a random Sudoku board
    # b = generate_random_board(difficulty=args.difficulty)                    # Set difficulty level to "easy", "medium", or "hard"
    # fig = visualize_board(b)
    # plt.show()

    # ## Solve the Sudoku board
    
    # solver.set_board(b.copy())
    # solver.solve()

    # ## Visualize the solved board
    # fig = visualize_board(board=solver.get_board(), initial_board=b)
    # plt.show()


    solver = SudokuSolver()

    ## CV Module
    path = args.image
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Failed to load the image. Check the file format.")
    
    plt.imshow(image)
    plt.show()
    plt.pause(2)


    ## Extracting Grid and Warping
    result = utils.preprocess(image)
    contours = utils.find_contours(result)
    contour = contours[0]
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    ordered_corners = utils.order_corners(approx)
    warped = utils.extract_sudoku_grid(image, contours[0])
    warped = utils.preprocess(warped)

    # Masking Lines and Extracting Cells
    masked = utils.mask_lines(warped)

    # Extracting Sudoku Board
    sudoku_matrix = utils.extract_sudoku_board(masked)
    sudoku_matrix = np.array(sudoku_matrix)
    fig=visualize_board(sudoku_matrix)
    plt.show()

    solver.set_board(sudoku_matrix.copy())

    if solver.is_valid():
        print("The board is valid")
        solver.solve()

        fig = visualize_board(board=solver.get_board(), initial_board=sudoku_matrix)
        plt.show()
        plt.pause(2)
    else:
        print("The board is not valid!\n Exitting...")


    
    

