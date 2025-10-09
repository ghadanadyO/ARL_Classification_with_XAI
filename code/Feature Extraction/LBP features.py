
#...............................................LBP features..............
import os
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from skimage.transform import resize
import numpy as np

# Define the root folder containing subfolders with images
root_folder =  './Dataset/Dataset 790'

# List of subfolder names (each containing images)
subfolders = os.listdir(root_folder)

# Initialize an empty DataFrame to store the features and a list for paths
features_df = pd.DataFrame()
image_paths = []

# Define LBP parameters
radius = 3
n_points = 24

# Loop through each subfolder
for subfolder in subfolders:
    subfolder_path = os.path.join(root_folder, subfolder)
    image_files = os.listdir(subfolder_path)
    # Loop through each image file in the subfolder
    for image_file in image_files:
        image_path = os.path.join(subfolder_path, image_file)
        # Handle image reading errors
        try:
            image = imread(image_path, as_gray=True)
        except:
            print(f"Error reading image: {image_path}")
            continue
        # Check image dimensions and resize if needed
        if image.shape != (128, 128):
            print(f"Resizing image: {image_path}")
            image = resize(image, (128, 128), anti_aliasing=True)
        # Calculate LBP features
        lbp_image = local_binary_pattern(image, n_points, radius, method="uniform")
        lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalize the histogram

        # Create a dictionary to store the features
        lbp_features = {f"LBP_{i}": hist_value for i, hist_value in enumerate(lbp_hist)}

        # Append the features and the image path to the DataFrame
        features_df = pd.concat([features_df, pd.DataFrame([lbp_features])], ignore_index=True)
        image_paths.append(image_path)  # Store the image path

# Add the image paths to the DataFrame
features_df['Image Path'] = image_paths

# Save the features DataFrame to an Excel file
excel_output_path = "./LBP_Features.xlsx"
features_df.to_excel(excel_output_path, index=False)



