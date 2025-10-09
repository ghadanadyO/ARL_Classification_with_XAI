#.......................GLCM features..................
import os
import pydicom
import numpy as np
import pandas as pd
from skimage import measure
from skimage.transform import resize
from skimage.feature import graycomatrix, graycoprops


# Function to perform image preprocessing
def preprocess_image(ct_image):
    resized_image = resize(ct_image, (128, 128), anti_aliasing=True)
    # Normalize the pixel values to [0, 1]
    normalized_image = resized_image / np.max(resized_image)
    # Convert to 8-bit unsigned integer
    normalized_image_uint8 = (normalized_image * 255).astype(np.uint8)
    return normalized_image

def extract_glcms_features(preprocessed_image):
    max_pixel_value = np.max(preprocessed_image)

    # Set levels to max_pixel_value + 1 to avoid the error
    levels = max_pixel_value + 1 if max_pixel_value < 256 else 256

    distances = [1, 2, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    glcms = [graycomatrix(preprocessed_image, distances, angles, levels=levels, symmetric=True, normed=True) for _ in
             range(3)]

    texture_features = {
        'GLCM_Contrast_Distance1': graycoprops(glcms[0], 'contrast').ravel(),
        'GLCM_Homogeneity_Distance1': graycoprops(glcms[0], 'homogeneity').ravel(),
        'GLCM_Energy_Distance1': graycoprops(glcms[0], 'energy').ravel(),
        'GLCM_Correlation_Distance1': graycoprops(glcms[0], 'correlation').ravel(),
        'GLCM_Contrast_Distance2': graycoprops(glcms[1], 'contrast').ravel(),
        'GLCM_Homogeneity_Distance2': graycoprops(glcms[1], 'homogeneity').ravel(),
        'GLCM_Energy_Distance2': graycoprops(glcms[1], 'energy').ravel(),
        'GLCM_Correlation_Distance2': graycoprops(glcms[1], 'correlation').ravel(),
        'GLCM_Contrast_Distance3': graycoprops(glcms[2], 'contrast').ravel(),
        'GLCM_Homogeneity_Distance3': graycoprops(glcms[2], 'homogeneity').ravel(),
        'GLCM_Energy_Distance3': graycoprops(glcms[2], 'energy').ravel(),
        'GLCM_Correlation_Distance3': graycoprops(glcms[2], 'correlation').ravel()
    }

    return texture_features


# Replace "root_folders" with a list of paths to the folders containing DICOM lung cancer images
root_folders = [
    "Dataset/Dataset 790",
    # Add more folder paths as needed
]

# Initialize lists to store image information and features
image_paths = []
texture_features_list = []

# Process each folder and extract texture-based features (GLCM)
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

                    # Extract texture-based features using GLCM
                    glcm_features = extract_glcms_features(preprocessed_image)

                    # Store the image information and features
                    image_paths.append(os.path.join(root, file))
                    texture_features_list.append(glcm_features)

                except Exception as e:
                    # Print an error message if loading or processing fails
                    print(f"Error processing {os.path.join(root, file)}: {e}")

# Create a DataFrame to store the results
df = pd.DataFrame({
    'Image Path': image_paths,
    **{f'{k}': [v.get(k) for v in texture_features_list] for k in texture_features_list[0]}
})

# Save the DataFrame to an Excel sheet
df.to_excel('./texture_features_glcm.xlsx', index=False)
