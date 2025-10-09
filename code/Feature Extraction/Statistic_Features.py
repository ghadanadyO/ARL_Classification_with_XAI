#.........................statistic Features ...................
import os
import pydicom
import numpy as np
import pandas as pd
import skimage.transform as skt
import numpy as np
from scipy.ndimage import zoom

def resize_ct_image(ct_image, scale_factor):
    if isinstance(scale_factor, (int, float)):
        # Convert to a tuple for all dimensions
        scale_factor = (scale_factor,) * 3
    # Ensure that the scale factor has three elements
    if len(scale_factor) != 3:
        raise ValueError("The scale_factor should be a float or a tuple with three elements.")
    # Perform the resizing using scipy.ndimage.zoom
    resized_image = zoom(ct_image, scale_factor, order=1)
    return resized_image


# Function to perform image preprocessing and feature extraction
def extract_intensity_features(ct_image):
    resized_image = skt.resize(ct_image, (128, 128), anti_aliasing=True)
    # Normalize the pixel values to [0, 1]
    normalized_image = resized_image / np.max(resized_image)
    # Calculate intensity-based features (statistical moments)
    mean_intensity = np.mean(normalized_image)
    std_intensity = np.std(normalized_image)
    median_intensity = np.median(normalized_image)
    skewness_intensity = np.mean((normalized_image - mean_intensity) ** 3) / (std_intensity ** 3)
    kurtosis_intensity = np.mean((normalized_image - mean_intensity) ** 4) / (std_intensity ** 4)
    return mean_intensity, std_intensity, median_intensity, skewness_intensity, kurtosis_intensity


# Replace "root_folders" with a list of paths to the folders containing DICOM lung cancer images
root_folders = [
    "./Dataset/Dataset 790",
]

# Initialize lists to store image information and features
image_paths = []
mean_intensities = []
std_intensities = []
median_intensities = []
skewness_intensities = []
kurtosis_intensities = []

# Process each folder and extract intensity-based features
for folder_path in root_folders:
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.dcm'):
                try:
                    # Read the DICOM file
                    dicom_data = pydicom.dcmread(os.path.join(root, file))

                    # Extract intensity-based features
                    mean_intensity, std_intensity, median_intensity, skewness_intensity, kurtosis_intensity = extract_intensity_features(
                        dicom_data.pixel_array)

                    # Store the image information and features
                    image_paths.append(os.path.join(root, file))
                    mean_intensities.append(mean_intensity)
                    std_intensities.append(std_intensity)
                    median_intensities.append(median_intensity)
                    skewness_intensities.append(skewness_intensity)
                    kurtosis_intensities.append(kurtosis_intensity)

                except Exception as e:
                    # Print an error message if loading or processing fails
                    print(f"Error processing {os.path.join(root, file)}: {e}")

# Create a DataFrame to store the results
data = {
    'Image Path': image_paths,
    'Mean Intensity': mean_intensities,
    'Std Intensity': std_intensities,
    'Median Intensity': median_intensities,
    'Skewness Intensity': skewness_intensities,
    'Kurtosis Intensity': kurtosis_intensities
}
df = pd.DataFrame(data)

# Save the DataFrame to an Excel sheet
df.to_excel('./statics_intensity_features.xlsx', index=False)

