# N-Queens Solver with Color Constraint

**Contributers:**
1. [Zeel Doshi](https://github.com/zeel-doshi)
2. [Jatin Pahuja](https://github.com/StrivingMind)
3. [Meenal Joshi]
4. [Sidharth]

### Overview:

This project is a Streamlit web application that solves the N-Queens puzzle with an added twist: color constraints. Users can upload an image of a colored N×N grid, and the application will find a valid solution where each queen is placed in a different colored region, adhering to the classic N-Queens rules.

### How It Works:

The application employs computer vision techniques to process the uploaded grid image and a backtracking algorithm to solve the constrained N-Queens problem. It offers two distinct approaches for image processing:

**Approach 1:** 

**Canny Edge Detection, Shi-Tomasi Corner Detection, and K-Means Clustering**

**Grid Detection:** This approach uses the Canny edge detector to find the outlines of the grid.

**Corner Detection:** The Shi-Tomasi corner detection algorithm is then used to precisely locate the corners of the cells in the grid.

**Cell Classification:** Each cell's color is classified using K-Means clustering to determine the dominant color and assign it to a specific region.


**Approach 2:**

**Harris Corner Detection, Sobel Edge Detection, and Gaussian Mixture Models (GMM)**

**Grid Detection:** This method utilizes the Sobel operator for edge detection.

**Corner Detection:** Harris corner detection is employed to identify the corners of the grid cells.

**Cell Classification:** A Gaussian Mixture Model (GMM) is used to classify the colors of the cells, which can be more robust for complex color variations.

### Constraint Satisfaction Problem (CSP) Solver

Once the grid and its color regions are identified, a backtracking algorithm solves the N-Queens problem. The core constraints are:

1. No two queens can be in the same row or column.

2. No two queens can be on the adjacent diagonal cells.

3. Color Constraint: Each queen must be placed in a unique color region.

The solver uses the Minimum Remaining Values (MRV) heuristic to prioritize placing queens in regions with the fewest available cells, which helps to prune the search space and find a solution more efficiently.

### Environment Setup

1. Clone the repository:
   ```bash
   git clone [repository_url]
   cd [repository_name]

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the streamlit app:
    ```bash
   streamlit run main.py

### How to Use web App

1. Open the Streamlit application in your web browser.

2. Select one of the two solving approaches from the dropdown menu.

3. Upload an image of a colored N×N grid. You can find sample grids at queens-game.com.

4. The application will process the image, display the detected grid and classified cells, and then show the final solution with queen icons placed on the board.

5. You can download the resulting image.


### Streamlit app:
https://n-queens-solver.streamlit.app/

### Use grids from:
https://www.queens-game.com/


