import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def read_image(file_path: str) -> np.ndarray:
    """Read an image from file."""
    image = cv2.imread(file_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {file_path}")
    return image

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale, blur it, and apply adaptive thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 3)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return binary

def find_largest_contour(image: np.ndarray) -> np.ndarray:
    """Find the largest contour in the image."""
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def extract_sudoku_grid(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Apply perspective transform to extract and warp the Sudoku grid."""
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) != 4:
        raise ValueError("The largest contour does not have 4 sides.")

    points = approx.reshape(4, 2)
    points = sorted(points, key=lambda x: x[1])  # Sort by y coordinate
    if points[0][0] > points[1][0]:
        points[0], points[1] = points[1], points[0]
    if points[2][0] > points[3][0]:
        points[2], points[3] = points[3], points[2]
    
    src = np.array(points, dtype="float32")
    side = max(
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[2] - points[3]),
        np.linalg.norm(points[0] - points[2]),
        np.linalg.norm(points[1] - points[3])
    )
    dst = np.array([[0, 0], [side - 1, 0], [0, side - 1], [side - 1, side - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (int(side), int(side)))
    return warped

def extract_digits_from_grid(grid_image: np.ndarray) -> np.ndarray:
    """Extract digits from the Sudoku grid."""
    cell_size = grid_image.shape[0] // 9
    digits = np.zeros((9, 9), dtype=int)

    for i in range(9):
        for j in range(9):
            cell = grid_image[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            digit = recognize_digit(cell)
            digits[i, j] = digit
    
    return digits

def recognize_digit(cell_image: np.ndarray) -> int:
    """Recognize a digit from a cell image."""
    # Placeholder function for digit recognition
    # In practice, this could use a pre-trained digit recognition model (e.g., CNN)
    return 0  # Assuming all cells are empty for now

if __name__ == "__main__":
    # Example usage
    image_path = "/home/screa/Mini-Projects/Sudoku_Solver/imgs/1.webp"
    image = read_image(image_path)
    plt.imshow(image)
    plt.show()
    preprocessed = preprocess_image(image)
    plt.imshow(preprocessed, cmap="gray")
    plt.show()
    largest_contour = find_largest_contour(preprocessed)
    sudoku_grid = extract_sudoku_grid(image, largest_contour)
    digits = extract_digits_from_grid(sudoku_grid)
    print(digits)
