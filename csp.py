import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#CSP
# Row column and adjacent diagonal constraint check
def is_safe(board, row, col):
    n = len(board)

    # Row and Column check
    if np.any(board[row, :]) or np.any(board[:, col]):
        return False

    # Adjacent Diagonal Check (only one step diagonals , as constraint only for adjacent diagonal)
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        r, c = row + dr, col + dc
        if 0 <= r < n and 0 <= c < n and board[r, c] == 1:
            return False

    return True

# valid cells for each region
def get_available_cells(region_matrix, board):
    """Returns a dict mapping region_id -> list of available (row, col)"""
    region_ids = np.unique(region_matrix)
    available = {region: [] for region in region_ids}
    n = len(region_matrix)

    for r in range(n):
        for c in range(n):
            if board[r, c] == 0 and is_safe(board, r, c):
                region = region_matrix[r, c]
                available[region].append((r, c))

    return available

# count of valid cells per region in sorted order
def print_available_summary(available_cells):
    """Utility to print the sorted available cell count summary per region."""
    print(".    Available cells per region:")
    sorted_regions = sorted(available_cells.items(), key=lambda item: len(item[1]))
    for region, cells in sorted_regions:
        print(f"  Region {region}: {len(cells)} cells")
    print("-" * 40)

#CSP + MRV
def solve_nqueens_csp(board, region_matrix, placed_regions):
    if len(placed_regions) == len(np.unique(region_matrix)):
        print("All queens successfully placed!\n", board)
        return True

    available_cells = get_available_cells(region_matrix, board)
    remaining_regions = [region for region in available_cells if region not in placed_regions]

    if not remaining_regions:
        return False

    print_available_summary(available_cells)

    # Sort remaining regions by number of available cells
    remaining_regions.sort(key=lambda r: len(available_cells[r]))

    current_region = remaining_regions[0] #pick the region with least available cells
    print(f"\nTrying region {current_region} with {len(available_cells[current_region])} available cells")


    for row, col in available_cells[current_region]:
        if is_safe(board, row, col):
            board[row, col] = 1
            placed_regions.add(current_region)
            print(f" Placed queen at ({row}, {col}) for region {current_region}")
            print("     Recomputing available cells after placement...")
            updated_available = get_available_cells(region_matrix, board) #update available cells
            #print_available_summary(updated_available)

            if solve_nqueens_csp(board, region_matrix, placed_regions):
                return True

            board[row, col] = 0
            placed_regions.remove(current_region) #backtrack
            print(f"   Backtracked from ({row}, {col}) for region {current_region}")

    return False


# n * n matrix to store placement of queens (1 denotes valid queen placement)
def nqueens_with_regions(region_matrix):
    n = len(region_matrix)
    board = np.zeros((n, n), dtype=int)
    placed_regions = set()

    if solve_nqueens_csp(board, region_matrix, placed_regions):
        return board
    else:
        print(" No valid solution found.")
        return None

# def overlay_icon(base_img, icon_img, position, size):
#     x, y = position
#     icon_resized = cv2.resize(icon_img, size)

#     # Check if icon has alpha channel (for transparency and opaque)
#     if icon_resized.shape[2] == 4: #if RGBA channels
#         icon_rgb = icon_resized[:, :, :3]
#         alpha_mask = icon_resized[:, :, 3] / 255.0
#     else:
#         icon_rgb = icon_resized
#         alpha_mask = np.ones((icon_resized.shape[0], icon_resized.shape[1]))

#     h, w = icon_rgb.shape[:2]
#     #blend icon with image
#     for c in range(3):
#         base_img[y:y+h, x:x+w, c] = (
#             alpha_mask * icon_rgb[:, :, c] +
#             (1 - alpha_mask) * base_img[y:y+h, x:x+w, c]
#         )
#     return base_img

# def place_queens(image_path, icon_path, board):
#     image = cv2.imread(image_path) #cropped input image
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED) #queen icon
#     n = len(board)
#     height, width = image.shape[:2]
#     cell_h = height // n
#     cell_w = width // n

#     #check for position of value 1 in board matrix and overlay queen on corresponding cell in original image
#     for i in range(n):
#         for j in range(n):
#             if board[i][j] == 1:
#                 x = j * cell_w
#                 y = i * cell_h
#                 image = overlay_icon(image, icon, (x, y), (cell_w, cell_h))

#     return image

def overlay_icon(base_img, icon_img, position, size):
    if icon_img is None:
        raise ValueError("❌ Queen icon image not loaded. Please check the icon path.")

    x, y = position
    icon_resized = cv2.resize(icon_img, size)

    # Check if icon has alpha channel (for transparency)
    if icon_resized.shape[2] == 4:  # RGBA
        icon_rgb = icon_resized[:, :, :3]
        alpha_mask = icon_resized[:, :, 3] / 255.0
    else:
        icon_rgb = icon_resized
        alpha_mask = np.ones((icon_resized.shape[0], icon_resized.shape[1]))

    h, w = icon_rgb.shape[:2]
    # Blend icon with base image
    for c in range(3):
        base_img[y:y+h, x:x+w, c] = (
            alpha_mask * icon_rgb[:, :, c] +
            (1 - alpha_mask) * base_img[y:y+h, x:x+w, c]
        )

    return base_img

def place_queens(image_path, icon_path, board):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("❌ Input board image not found. Please check the image path.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
    if icon is None:
        raise ValueError("❌ Queen icon image not found. Please check the icon path.")

    n = len(board)
    height, width = image.shape[:2]
    cell_h = height // n
    cell_w = width // n

    for i in range(n):
        for j in range(n):
            if board[i][j] == 1:
                x = j * cell_w
                y = i * cell_h
                image = overlay_icon(image, icon, (x, y), (cell_w, cell_h))

    return image