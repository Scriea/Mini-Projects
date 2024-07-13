import os
import sys
sys.path.append(os.path.dirname(os.path.abspath('.')))
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract

# from model import predict_digit


def preprocess(img):
    """
    Preprocess the image for board extraction.
    1. Convert to grayscale
    2. Apply Gaussian blur : For noise removal
    3. Adaptive thresholding : For better edge detection
    4. Morphological operations : For enhanced noise removal and better contour detection
    """
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    blur = cv2.GaussianBlur(img_gray, (9, 9), 0)

    ## Adaptive thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    inverted = cv2.bitwise_not(thresh, 0)
    
    ## Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
    result = cv2.dilate(morph, kernel, iterations=1)
    return result

def find_contours(image):
    """
    Find the contours in the preprocessed image.
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours

def draw_contours(image, contour):
    """
    Draw the contours on the image.
    """
    img_copy = image.copy()
    cv2.drawContours(img_copy, [contour], 0, (0, 255, 0), 3)
    return img_copy

def order_corners(corners):
    """
    Sort the points based on their x and y coordinates. 
    It orders the points in the following order: top-left, top-right, bottom-right, bottom-left.
    """
    corners = np.array([point[0] for point in corners], dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum, and the bottom-right point will have the largest sum.
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]

    # The top-right point will have the smallest difference and the bottom-left will have the largest difference.
    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]
    return rect

def extract_sudoku_grid(image, contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    # Check if the contour has four corners
    if len(approx) == 4:  
        rect = order_corners(approx)                                    
        (top_left, top_right, bottom_right, bottom_left) = rect
        # top_left, top_right, bottom_right, bottom_left = corners
        width = int(max(np.linalg.norm(top_right - top_left),
                        np.linalg.norm(bottom_right - bottom_left)))
        height = int(max(np.linalg.norm(top_right - bottom_right),
                         np.linalg.norm(top_left - bottom_left)))

        destination_points = np.float32([[0, 0], [width - 1, 0],
                                         [width - 1, height - 1], [0, height - 1]])
        transform_matrix = cv2.getPerspectiveTransform(rect, destination_points)
        unwrapped = cv2.warpPerspective(image, transform_matrix, (width, height))
        return unwrapped
    else:
        raise Exception("The largest contour does not have four corners.")


def draw_lines(img, lines):
    clone = img.copy()
    lines = np.squeeze(lines)

    for rho, theta in lines:
        # find out where the line stretches to and draw them
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x0 = cos_t * rho
        y0 = sin_t * rho
        x1 = int(x0 + 1000 * (-sin_t))
        y1 = int(y0 + 1000 * cos_t)
        x2 = int(x0 - 1000 * (-sin_t))
        y2 = int(y0 - 1000 * cos_t)
        cv2.line(clone, (x1, y1), (x2, y2), (255, 255, 255), 4)
    return clone

def mask_lines(image):
    """
    Mask the horizontal and vertical lines of the Sudoku grid.
    """
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(image.shape[1] / 10), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(image.shape[0] / 10)))

    # Detect horizontal and vertical lines
    horizontal_lines = cv2.erode(image, horizontal_kernel)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel)

    vertical_lines = cv2.erode(image, vertical_kernel)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel)
    # Combine line images
    grid_lines = cv2.add(horizontal_lines, vertical_lines)
    grid = cv2.adaptiveThreshold(grid_lines, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 235, 2)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=2)

    # find the list of where the lines are, this is an array of (rho, theta in radians)
    pts = cv2.HoughLines(grid, .3, np.pi / 90, 200)

    lines = draw_lines(grid, pts)
    # extract the lines so only the numbers remain
    mask = cv2.bitwise_not(lines)
    masked_img = cv2.bitwise_and(image, mask)
    return masked_img


def clean_cell(img, k=0.4, l=0.4):
    height, width = img.shape
    midW, midH = width // 2, height//2
    x_start, x_end = int(midH-height*l), int(midH+height*l)
    y_start, y_end  = int(midW - width * k), int(midW + width * k)
    
    mid_img = img[x_start:x_end, y_start:y_end]

    ## Fraction of 0 pixels i.e blank space in sudoku board
    frac_zero = np.isclose(mid_img, 0).sum() / ((x_end-x_start)*(y_end-y_start)) 
    
    if frac_zero >= 0.9:
        return np.zeros_like(img), False


    # center image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(contours[0])

    start_x = (width - w) // 2
    start_y = (height - h) // 2
    new_img = np.zeros_like(img)
    new_img[start_y:start_y + h, start_x:start_x + w] = img[y:y + h, x:x + w]
    return new_img, True


def extract_digit_img(image):
    """
    Extracts a single digit from a provided grayscale image using classical image processing techniques with OpenCV.

    :param gray_image: Grayscale image containing a single digit.
    :return: Cropped image of the digit or None if no valid digit is found.
    """
    
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and find the digit contour
    digit_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:  # Assuming the largest contour in a single digit image is the digit
            max_area = area
            digit_contour = contour

    if digit_contour is None:
        return None

    # Compute the bounding rectangle for the digit contour
    x, y, w, h = cv2.boundingRect(digit_contour)
    
    # Crop and return the image around the digit
    digit_image = image[y:y+h, x:x+w]
    return digit_image


## Was performing really bad
def digit_OCR(img):
    """
    Extracts a digit from a provided grayscale image using OCR with Tesseract.
    """
    # Convert the image to a binary form
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Extract the digit using OCR
    digit = pytesseract.image_to_string(thresh, config='--psm 10 --oem 3 -c tessedit_char_whitelist=123456789').strip()
    digit = int(digit) if digit else 0
    return digit


def extract_sudoku_board(image, margin=5, debug=False):
    """
    Divide the grid image into 81 individual cells and extract digits from each cell.
    Returns a 2D array representing the Sudoku grid.
    """
    cell_height, cell_width = image.shape[0] // 9, image.shape[1] // 9
    sudoku_grid = []

    for y in range(9):
        row = []
        for x in range(9):
            x_start = max(0, x * cell_width - margin)
            x_end = min(image.shape[1], (x + 1) * cell_width + margin)
            y_start = max(0, y * cell_height - margin)
            y_end = min(image.shape[0], (y + 1) * cell_height + margin)
            cell = image[y_start:y_end, x_start:x_end]
            

            cell, valid = clean_cell(cell)
            if debug: 
                plt.imshow(cell)
                plt.show()
            
            if valid:
                cell = cv2.resize(cell, (28, 28))
                # digit = predict_digit(cell)
                docr = digit_OCR(cell)
                row.append(docr)  
            else:
                row.append(0)

        sudoku_grid.append(row)

    return sudoku_grid


if __name__ =="__main__":
    pass
    ## Read Image
    # img_path = "/home/screa/Desktop/Mini-Projects/Sudoku_Solver/imgs/sudoku.jpg"
    # image = cv2.imread(img_path)
    # if image is None:
    #     raise ValueError("Failed to load the image. Check the file format.")
    
    # ## Extracting Grid and Warping
    # result = preprocess(image)
    # contours = find_contours(result)
    # contour = contours[0]
    # peri = cv2.arcLength(contour, True)
    # approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    # ordered_corners = order_corners(approx)
    # warped = extract_sudoku_grid(image, contours[0])
    # warped = preprocess(warped)

    # # Masking Lines and Extracting Cells
    # masked = mask_lines(warped)

    # # Extracting Sudoku Board
    # sudoku_matrix = extract_sudoku_board(masked, debug=False)
    # print(np.array(sudoku_matrix))