import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def save_and_show_image_1(image, filename, cmap=None):
    """Save the image and display it."""
    cv2.imwrite(filename, image)
    return image

def process_image(image_path):
    """Apply Canny edge detection and crop the image."""
    color_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 150)

    # Find bounding box
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        x1, y1, x2, y2 = x, y, x + w, y + h
    else:
        x1, y1, x2, y2 = 0, 0, edges.shape[1], edges.shape[0]

    # Crop images
    cropped_color = color_image[y1:y2, x1:x2]
    cropped_blurred = blurred_image[y1:y2, x1:x2]
    cropped_edges = edges[y1:y2, x1:x2]

    # Save images
    save_and_show_image_1(cropped_color, "cropped_color.png")
    save_and_show_image_1(cropped_blurred, "cropped_blurred.png", cmap='gray')
    save_and_show_image_1(cropped_edges, "cropped_edges.png", cmap='gray')

    # Display images horizontally
    plt.figure(figsize=(15, 5))
    images = [cropped_color, cropped_blurred, cropped_edges]
    titles = ["Cropped Original", "Cropped Blurred", "Cropped Edges"]
    cmaps = [None, "gray", "gray"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[i] if cmaps[i] is None else images[i], cmap=cmaps[i])
        plt.title(titles[i])
        plt.axis("off")

    plt.show()
    return cropped_color, cropped_blurred, cropped_edges

def process_image_a2(image_path):
    """Apply Sobel edge detection and crop the image."""
    color_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Sobel edge detection
    sobelx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.magnitude(sobelx, sobely)
    sobel_edges = np.uint8(np.clip(sobel_edges, 0, 255))  # Convert to 8-bit image

    # Find bounding box
    contours, _ = cv2.findContours(sobel_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        x1, y1, x2, y2 = x, y, x + w, y + h
    else:
        x1, y1, x2, y2 = 0, 0, sobel_edges.shape[1], sobel_edges.shape[0]

    # Crop images
    cropped_color = color_image[y1:y2, x1:x2]
    cropped_blurred = blurred_image[y1:y2, x1:x2]
    cropped_edges = sobel_edges[y1:y2, x1:x2]

    # Save images
    save_and_show_image_1(cropped_color, "cropped_color.png")
    save_and_show_image_1(cropped_blurred, "cropped_blurred.png", cmap='gray')
    save_and_show_image_1(cropped_edges, "cropped_edges.png", cmap='gray')

    # Display images horizontally
    plt.figure(figsize=(15, 5))
    images = [cropped_color, cropped_blurred, cropped_edges]
    titles = ["Cropped Original", "Cropped Blurred", "Cropped Sobel Edges"]
    cmaps = [None, "gray", "gray"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[i] if cmaps[i] is None else images[i], cmap=cmaps[i])
        plt.title(titles[i])
        plt.axis("off")

    plt.show()
    return cropped_color, cropped_blurred, cropped_edges

def save_and_show_image(image, filename):
    """Save and display the image."""
    cv2.imwrite(filename, image)
    plt.imshow(image, cmap="gray")  # Since it's already grayscale
    plt.axis("off")
    plt.show()

def detect_and_refine_corners(image_path, max_corners=100, quality_level=0.01, min_distance=10):
    """Detect and refine corners using Shi-Tomasi and sub-pixel refinement."""
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

    # Detect strong corners using Shi-Tomasi method
    corners = cv2.goodFeaturesToTrack(gray_image, maxCorners=max_corners,
                                      qualityLevel=quality_level, minDistance=min_distance)

    if corners is not None:
        corners = corners.astype(int)  # Convert to integer format

        # Refine corner locations to sub-pixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
        corners = cv2.cornerSubPix(gray_image, np.float32(corners), (5, 5), (-1, -1), criteria)

        # Draw refined corners
        color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)  # Convert for visualization
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(color_image, (int(x), int(y)), 5, (0, 0, 255), -1)

        # Save and display results
        save_and_show_image(color_image, "refined_corners.png")

        return corners

    return None

def detect_and_refine_corners_a2(image_path, max_corners=100, quality_level=0.01, min_distance=10, harris_k=0.04):
    """Detect and refine corners using Harris and sub-pixel refinement."""
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Detect corners using Harris corner detection
    corners = cv2.goodFeaturesToTrack(gray_image, maxCorners=max_corners,
                                      qualityLevel=quality_level, minDistance=min_distance,
                                      useHarrisDetector=True, k=harris_k)

    if corners is not None:
        corners = corners.astype(int)  # Convert to integer for initial drawing (optional)

        # Refine corners to sub-pixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
        corners = cv2.cornerSubPix(gray_image, np.float32(corners), (5, 5), (-1, -1), criteria)

        # Draw refined corners
        color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(color_image, (int(x), int(y)), 5, (0, 0, 255), -1)

        save_and_show_image(color_image, "harris_refined_corners.png")
        return corners

    return None

def draw_grid(image_path, corners):
    """Draws a grid and saves the result with lines and corners."""
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if gray is None:
        raise ValueError("Image not found or unable to load.")

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    line_count = 0
    if lines is not None:
        line_count = len(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

    output_path = "grid_overlay.png"
    cv2.imwrite(output_path, image)

    print(f"Number of detected lines: {line_count}")
    return output_path


    # Print number of detected lines
    print(f"Number of detected lines: {line_count}")

def organize_corners_dynamically(corners):
    """Dynamically organize detected corners into a row-wise list."""
    corners = sorted(corners.tolist(), key=lambda x: (x[0][1], x[0][0]))  # Sort by Y, then X

    rows = []
    current_row = [corners[0]]  # Start with the first corner
    row_threshold = 10  # Adjust if needed (based on Y-coordinate differences)

    for i in range(1, len(corners)):
        if abs(corners[i][0][1] - current_row[-1][0][1]) > row_threshold:
            # New row detected
            rows.append(sorted(current_row, key=lambda p: p[0][0]))  # Sort row by X
            current_row = [corners[i]]
        else:
            current_row.append(corners[i])

    if current_row:
        rows.append(sorted(current_row, key=lambda p: p[0][0]))  # Add last row

    # Convert to (x, y) format **without rounding to int**
    structured_corners = [[[x, y] for [[x, y]] in row] for row in rows]

    return structured_corners


def organize_corners_cellwise(corners):
    """Organize detected corners into row-wise cell structure (each cell has 4 corners)."""
    structured_corners = organize_corners_dynamically(corners)  # First, get row-wise sorted points

    grid_structure = []  # This will store the rows of cells

    for row_idx in range(len(structured_corners) - 1):  # Loop through rows (except last row)
        row_cells = []  # Store cells for the current row

        for col_idx in range(len(structured_corners[row_idx]) - 1):  # Loop through columns
            # Extract 4 corners for each cell
            top_left = structured_corners[row_idx][col_idx]
            top_right = structured_corners[row_idx][col_idx + 1]
            bottom_left = structured_corners[row_idx + 1][col_idx]
            bottom_right = structured_corners[row_idx + 1][col_idx + 1]

            row_cells.append([top_left, top_right, bottom_left, bottom_right])

        grid_structure.append(row_cells)  # Append the row of cells

    return grid_structure  # Now it's a list of rows, each row is a list of cells

def store_cells_in_dict(corners):
    """Store cell-wise corner points in a dictionary with keys as 'row_col'."""
    structured_corners = organize_corners_dynamically(corners)  # Get sorted row-wise points
    cell_dict = {}  # Dictionary to store cell-wise corner points

    for row_idx in range(len(structured_corners) - 1):  # Iterate over rows (except last row)
        for col_idx in range(len(structured_corners[row_idx]) - 1):  # Iterate over columns
            # Extract 4 corners for each cell
            top_left = structured_corners[row_idx][col_idx]
            top_right = structured_corners[row_idx][col_idx + 1]
            bottom_left = structured_corners[row_idx + 1][col_idx]
            bottom_right = structured_corners[row_idx + 1][col_idx + 1]

            # Store in dictionary using 'row_col' as the key
            cell_dict[f"{row_idx}_{col_idx}"] = [top_left, top_right, bottom_left, bottom_right]

    return cell_dict

def extract_and_classify_cells(image_path, cell_dict, crop_ratio=0.70):
    """Extracts inner regions of each cell, classifies them based on dominant color, and visualizes bounding boxes."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    class_colors = {}
    cell_classes = {}

    for key, corners in cell_dict.items():
        # Convert float coordinates to integers
        corners = np.array(corners, dtype=np.int32)

        # Get bounding box coordinates
        x_min = min(corners[:, 0])
        x_max = max(corners[:, 0])
        y_min = min(corners[:, 1])
        y_max = max(corners[:, 1])

        # Compute reduced bounding box size (to avoid edges)
        width = x_max - x_min
        height = y_max - y_min
        x_min_new = int(x_min + (1 - crop_ratio) * width / 2)
        x_max_new = int(x_max - (1 - crop_ratio) * width / 2)
        y_min_new = int(y_min + (1 - crop_ratio) * height / 2)
        y_max_new = int(y_max - (1 - crop_ratio) * height / 2)

        # Extract cropped cell region
        cell_region = image[y_min_new:y_max_new, x_min_new:x_max_new]

        # Resize for KMeans clustering
        reshaped_region = cell_region.reshape(-1, 3)

        # Perform K-Means clustering to find dominant color
        kmeans = KMeans(n_clusters=1, random_state=0, n_init=10)
        kmeans.fit(reshaped_region)
        dominant_color = tuple(map(int, kmeans.cluster_centers_[0]))

        # Assign a class ID based on unique colors
        if dominant_color not in class_colors:
            class_colors[dominant_color] = len(class_colors) + 1
        cell_class = class_colors[dominant_color]
        cell_classes[key] = cell_class

        # Draw bounding box and class label
        cv2.rectangle(image, (x_min_new, y_min_new), (x_max_new, y_max_new), (0, 255, 0), 2)
        cv2.putText(image, str(cell_class), (x_min_new + 5, y_min_new + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Save and display the classified image
    cv2.imwrite("classified_cells.png", image)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    return cell_classes

def extract_and_classify_cells_a2(image_path, cell_dict, crop_ratio=0.75):
    """Extracts inner regions of each cell, classifies them based on dominant color using GMM, and visualizes bounding boxes."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    class_colors = {}
    cell_classes = {}
    gmm_means = []
    gmm_covariances = []
    keys = []

    for key, corners in cell_dict.items():
        corners = np.array(corners, dtype=np.int32)

        # Bounding box
        x_min = min(corners[:, 0])
        x_max = max(corners[:, 0])
        y_min = min(corners[:, 1])
        y_max = max(corners[:, 1])

        # Crop to inner region
        width = x_max - x_min
        height = y_max - y_min
        x_min_new = int(x_min + (1 - crop_ratio) * width / 2)
        x_max_new = int(x_max - (1 - crop_ratio) * width / 2)
        y_min_new = int(y_min + (1 - crop_ratio) * height / 2)
        y_max_new = int(y_max - (1 - crop_ratio) * height / 2)

        cell_region = image[y_min_new:y_max_new, x_min_new:x_max_new]
        reshaped_region = cell_region.reshape(-1, 3)

        # Use GMM to find dominant color
        gmm = GaussianMixture(n_components=1, random_state=0)
        gmm.fit(reshaped_region)
        dominant_color = tuple(map(int, gmm.means_[0]))


        gmm_means.append(gmm.means_[0])
        gmm_covariances.append(gmm.covariances_[0])
        keys.append(key)


        if dominant_color not in class_colors:
            class_colors[dominant_color] = len(class_colors) + 1
        cell_class = class_colors[dominant_color]
        cell_classes[key] = cell_class

        # Draw bounding box and label
        cv2.rectangle(image, (x_min_new, y_min_new), (x_max_new, y_max_new), (0, 255, 0), 2)
        cv2.putText(image, str(cell_class), (x_min_new + 5, y_min_new + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Save and show result
    cv2.imwrite("classified_cells_gmm.png", image)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    return cell_classes, gmm_means, gmm_covariances, keys


def convert_dict_to_numpy(cell_classes):
    """Convert cell_classes dictionary to a NumPy array."""
    # Determine the grid size from the keys (assumes keys are formatted as 'row_col')
    rows, cols = zip(*[map(int, key.split('_')) for key in cell_classes.keys()])
    grid_size = (max(rows) + 1, max(cols) + 1)  # Shape of the grid

    # Initialize a NumPy array with -1 (assuming -1 represents unknown class)
    class_array = np.full(grid_size, -1, dtype=int)

    # Populate the array using dictionary values
    for key, class_value in cell_classes.items():
        row, col = map(int, key.split('_'))
        class_array[row, col] = class_value

    return class_array


