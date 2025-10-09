import pandas as pd
from sklearn.preprocessing import StandardScaler

# Replace "file_path" with the path to your Excel file containing the feature data
file_path = "./all_features.xlsx"

# Read the Excel file into a DataFrame
df = pd.read_excel(file_path)

# Specify the feature columns you want to normalize
selected_features = ['GLRLM_contrast_1', 'GLRLM_dissimilarity_1', 'GLRLM_homogeneity_1', 'GLRLM_energy_1',
                   'GLRLM_correlation_1', 'GLRLM_contrast_2', 'GLRLM_dissimilarity_2', 'GLRLM_homogeneity_2',
                   'GLRLM_energy_2', 'GLRLM_correlation_2', 'GLRLM_contrast_3', 'GLRLM_dissimilarity_3',
                   'GLRLM_homogeneity_3', 'GLRLM_energy_3', 'GLRLM_correlation_3',
                   'Mean Intensity', 'Std Intensity', 'Median Intensity', 'Skewness Intensity', 'Kurtosis Intensity',
                    'Areas', 'Perimeters', 'Compactnesses',
                   'LBP_0', 'LBP_1', 'LBP_2', 'LBP_3','LBP_4', 'LBP_5', 'LBP_6', 'LBP_7', 'LBP_8', 'LBP_9',
                    'LBP_10', 'LBP_11', 'LBP_12', 'LBP_13', 'LBP_14',
                   'LBP_15', 'LBP_16', 'LBP_17', 'LBP_18', 'LBP_19', 'LBP_20', 'LBP_21', 'LBP_22', 'LBP_23', 'LBP_24',
                   'LBP_25']

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the selected features and transform the data
normalized_features = scaler.fit_transform(df[selected_features])

# Create a DataFrame with normalized features
df_normalized = pd.DataFrame(normalized_features, columns=selected_features)

# Save the normalized DataFrame to a new Excel file
output_file_path = "./all_features_Normalization2.xlsx"
df_normalized.to_excel(output_file_path, index=False)
