import streamlit as st
import numpy as np
import cv2
from PIL import Image
from preprocessing import *
from csp import *

st.set_page_config(page_title="N-Queens Solver", layout="centered")

st.title("♟️ N-Queens with Color Constraints")
st.markdown("Upload a colored N×N grid image. We'll solve the N-Queens puzzle with region constraints.")

# 🔽 Dropdown menu for approach selection
approach = st.selectbox(
    "Choose solving approach:",
    ("Approach 1: Canny + Shi-Tomasi + KMeans", "Approach 2: Harris + Sobel + GMM")
)

# 🖼 Upload section
uploaded_file = st.file_uploader("Upload a grid image", type=["png", "jpg", "jpeg"])

# ✅ Proceed if Approach 1 is selected
if approach == "Approach 1: Canny + Shi-Tomasi + KMeans" and uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    cv2_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    input_path = "input_image.png"
    cv2.imwrite(input_path, cv2_image)

    st.image(image_np, caption="📤 Uploaded Image", use_container_width=True)
    st.info("🔄 Processing image...")

    try:
        # Step 1: Crop
        process_image(input_path)

        # Step 2: Corners
        corners = detect_and_refine_corners("cropped_blurred.png", max_corners=100)
        if corners is None:
            st.error("❌ Could not detect corners. Try a cleaner image.")
            st.stop()

        # Step 3: Store cells
        cell_coordinates_dict = store_cells_in_dict(corners)

        # Step 4: Classify
        cell_classes = extract_and_classify_cells("cropped_color.png", cell_coordinates_dict)
        region_matrix = convert_dict_to_numpy(cell_classes)
        print(region_matrix)
        # 🎨 Show intermediate stages
        st.markdown("### 🧭 Grid Detection Process")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image("refined_corners.png", caption="🎯 Refined Corners", use_container_width=True)
        with col2:
            grid_overlay_path = draw_grid("cropped_blurred.png", corners)
            st.image(grid_overlay_path, caption="📐 Grid with Lines & Corners", use_container_width=True)
        with col3:
            st.image("classified_cells.png", caption="🧩 Classified Cells", use_container_width=True)
        
        # Step 5: CSP Solver
        st.info("🧩 Solving N-Queens with region constraints...")
        solution = nqueens_with_regions(region_matrix)

        if solution is None:
            st.error("❌ No valid solution found.")
        else:
            final_img = place_queens("cropped_color.png", "queen_icon.png", solution)
            result_path = "Nqueen_result.png"
            save_and_show_image_1(final_img, result_path)

            st.success("✅ Solution found!")
            st.image(final_img, caption="♛ N-Queens Solution", use_container_width=True)

            with open(result_path, "rb") as file:
                st.download_button(
                    label="📥 Download Result",
                    data=file,
                    file_name="Nqueen_result.png",
                    mime="image/png"
                )

    except Exception as e:
        st.error(f"Something went wrong: {e}")

elif approach == "Approach 2: Harris + Sobel + GMM" and uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    cv2_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    input_path = "input_image.png"
    cv2.imwrite(input_path, cv2_image)

    st.image(image_np, caption="📤 Uploaded Image", use_container_width=True)
    st.info("🔄 Processing image...")

    try:
        # Step 1: Crop
        process_image_a2(input_path)

        # Step 2: Corners
        corners = detect_and_refine_corners_a2("cropped_blurred.png", max_corners=100)
        if corners is None:
            st.error("❌ Could not detect corners. Try a cleaner image.")
            st.stop()

        # Step 3: Store cells
        cell_coordinates_dict = store_cells_in_dict(corners)

        # Step 4: Classify
        cell_classes, gmm_means, gmm_covariances, keys = extract_and_classify_cells_a2("cropped_color.png", cell_coordinates_dict)
        region_matrix = convert_dict_to_numpy(cell_classes)
        print(region_matrix)
        # 🎨 Show intermediate stages
        st.markdown("### 🧭 Grid Detection Process")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image("harris_refined_corners.png", caption="🎯 Refined Corners", use_container_width=True)
        with col2:
            grid_overlay_path = draw_grid("cropped_blurred.png", corners)
            st.image(grid_overlay_path, caption="📐 Grid with Lines & Corners", use_container_width=True)
        with col3:
            st.image("classified_cells_gmm.png", caption="🧩 Classified Cells", use_container_width=True)
        
        # Step 5: CSP Solver
        st.info("🧩 Solving N-Queens with region constraints...")
        solution = nqueens_with_regions(region_matrix)

        if solution is None:
            st.error("❌ No valid solution found.")
        else:
            final_img = place_queens("cropped_color.png", "queen_icon.png", solution)
            result_path = "Nqueen_result.png"
            save_and_show_image_1(final_img, result_path)

            st.success("✅ Solution found!")
            st.image(final_img, caption="♛ N-Queens Solution", use_container_width=True)

            with open(result_path, "rb") as file:
                st.download_button(
                    label="📥 Download Result",
                    data=file,
                    file_name="Nqueen_result.png",
                    mime="image/png"
                )

    except Exception as e:
        st.error(f"Something went wrong: {e}")