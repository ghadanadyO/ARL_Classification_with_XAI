import os
import pydicom
import numpy as np
import pandas as pd
from skimage import measure
from skimage.transform import resize
from skimage import filters

def simple_image_segmentation(image):
    # Apply an appropriate thresholding method, such as Otsu's thresholding
    threshold_value = filters.threshold_otsu(image)

    # Create a binary mask based on the threshold value
    binary_mask = image > threshold_value

    return binary_mask

# Function to perform image preprocessing
def preprocess_image(ct_image):
 
    resized_image = resize(ct_image, (128, 128), anti_aliasing=True)

    # Normalize the pixel values to [0, 1]
    normalized_image = resized_image / np.max(resized_image)

    return normalized_image

# Function to extract shape-based features from binary mask
def extract_shape_features(binary_mask):
    # Compute region properties using skimage.measure.regionprops
    props = measure.regionprops(binary_mask.astype(int))  # Replaced np.int with int

    # Initialize feature lists
    areas = []
    perimeters = []
    compactnesses = []

    # Extract shape-based features from each region
    for prop in props:
        areas.append(prop.area)
        perimeters.append(prop.perimeter)
        compactnesses.append(4 * np.pi * prop.area / (prop.perimeter ** 2))

    return areas, perimeters, compactnesses

# Replace "root_folders" with a list of paths to the folders containing DICOM lung cancer images
root_folders = [
    './Dataset/Dataset 790',
]

# Initialize lists to store image information and features
image_paths = []
areas_list = []
perimeters_list = []
compactnesses_list = []

# Process each folder and extract shape-based features
for folder_path in root_folders:
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.dcm'):
                try:
                    # Read the DICOM file
                    dicom_data = pydicom.dcmread(os.path.join(root, file))
                    ct_image = dicom_data.pixel_array

                    # Preprocess the CT image
                    preprocessed_image = preprocess_image(ct_image)

                    # Perform image segmentation to obtain a binary mask
                    # Replace the following segmentation method with your own segmentation approach
                    binary_mask = simple_image_segmentation(preprocessed_image)

                    # Extract shape-based features
                    areas, perimeters, compactnesses = extract_shape_features(binary_mask)

                    # Store the image information and features
                    image_paths.append(os.path.join(root, file))
                    areas_list.append(areas)
                    perimeters_list.append(perimeters)
                    compactnesses_list.append(compactnesses)

                except Exception as e:
                    # Print an error message if loading or processing fails
                    print(f"Error processing {os.path.join(root, file)}: {e}")

# Create a DataFrame to store the results
data = {
    'Image Path': image_paths,
    'Areas': areas_list,
    'Perimeters': perimeters_list,
    'Compactnesses': compactnesses_list
}
df = pd.DataFrame(data)

# Save the DataFrame to an Excel sheet
df.to_excel('./shape_features.xlsx', index=False)


