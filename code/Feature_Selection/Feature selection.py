
#
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

# Load your data from the Excel file into a DataFrame
df = pd.read_excel("./_all_features_Normalization.xlsx")

# Assume your target variable is named 'target_column'
X = df.drop(columns=['Cancer'])
y = df['Cancer']

# Specify the number of features to select
k = 20 # You can change this value based on your requirement

# Perform Correlation-based Feature Selection (SelectKBest)
k_best = SelectKBest(score_func=f_regression, k=k)
selected_features = k_best.fit_transform(X, y)

# Get the indices of the selected features
selected_feature_indices = k_best.get_support(indices=True)

# Create a DataFrame with selected features and target variable
selected_features_df = df.iloc[:, selected_feature_indices].copy()
selected_features_df['Cancer'] = y

# Save the selected features and target variable to a new Excel file
output_file_path = "./Correlation-based Feature Selection (CFS).xlsx"
selected_features_df.to_excel(output_file_path, index=False)
#........................... RFE Feature selection.........................
#
# import pandas as pd
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LinearRegression
#
# # Load your data from the Excel file into a DataFrame
# df = pd.read_excel("./all_features_Normalization.xlsx")
#
# # Assume your target variable is named 'target_column'
# X = df.drop(columns=['Cancer'])
# y = df['Cancer']
#
# # Specify the number of features to select
# n_features = 20 # You can change this value based on your requirement
#
# # Initialize the RFE model with a base estimator (Linear Regression in this case)
# model_rfe = LinearRegression()
# rfe = RFE(model_rfe, n_features_to_select=n_features)
#
# # Perform Recursive Feature Elimination
# selected_features = rfe.fit_transform(X, y)
#
# # Get the indices of the selected features
# selected_feature_indices = rfe.get_support(indices=True)
#
# # Create a DataFrame with selected features and target variable
# selected_features_df = df.iloc[:, selected_feature_indices].copy()
# selected_features_df['Cancer'] = y
#
# # Save the selected features and target variable to a new Excel file
# output_file_path = "./features_selected_RFE.xlsx"
# selected_features_df.to_excel(output_file_path, index=False)

#.........................Random Forest Feature Importance............
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
#
# # Load your data from the Excel file into a DataFrame
# df = pd.read_excel("./all_features_Normalization.xlsx")
#
# # Assume your target variable is named 'target_column'
# X = df.drop(columns=['Cancer'])
# y = df['Cancer']
#
# # Initialize the Random Forest model
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
#
# # Fit the Random Forest model to your data
# rf_model.fit(X, y)
#
# # Get feature importances from the model
# feature_importances = rf_model.feature_importances_
#
# # Sort features based on importance
# sorted_indices = feature_importances.argsort()[::-1]
#
# # Specify the number of features to select
# n_features = 20  # You can adjust this based on your requirement
#
# # Get the indices of the top features
# selected_feature_indices = sorted_indices[:n_features]
#
# # Create a DataFrame with selected features and target variable
# selected_features_df = df.iloc[:, selected_feature_indices]
# selected_features_df = selected_features_df.copy()
# selected_features_df['Cancer'] = y
#
# # Save the selected features and target variable to a new Excel file
# output_file_path = "./features_selected_Forest_Feature.xlsx"
# selected_features_df.to_excel(output_file_path, index=False)
#


