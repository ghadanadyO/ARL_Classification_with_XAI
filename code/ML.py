import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB  # For Bayesian Network
from xgboost import XGBClassifier  # For XGBoost

df = pd.read_excel("./All_Features_Normalization2.xlsx")


feature_columns = [
    'GLRLM_contrast_1', 'GLRLM_dissimilarity_1', 'GLRLM_homogeneity_1', 'GLRLM_energy_1',
    'GLRLM_correlation_1', 'GLRLM_contrast_2', 'GLRLM_dissimilarity_2', 'GLRLM_homogeneity_2',
    'GLRLM_energy_2', 'GLRLM_correlation_2', 'GLRLM_contrast_3', 'GLRLM_dissimilarity_3',
    'GLRLM_homogeneity_3', 'GLRLM_energy_3', 'GLRLM_correlation_3',
    'Mean Intensity', 'Std Intensity', 'Median Intensity', 'Skewness Intensity', 'Kurtosis Intensity',
    'Areas', 'Perimeters', 'Compactnesses',
    'LBP_0', 'LBP_1', 'LBP_2', 'LBP_3', 'LBP_4', 'LBP_5', 'LBP_6', 'LBP_7', 'LBP_8', 'LBP_9',
    'LBP_10', 'LBP_11', 'LBP_12', 'LBP_13', 'LBP_14', 'LBP_15', 'LBP_16', 'LBP_17', 'LBP_18',
    'LBP_19', 'LBP_20', 'LBP_21', 'LBP_22', 'LBP_23', 'LBP_24', 'LBP_25'
]
for col in feature_columns:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df[col] = df[col].apply(lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) else x)

X = df[feature_columns]
y = df['Cancer']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Bayesian Network": GaussianNB(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = []

for name, clf in classifiers.items():
    print(f"\n🔹 Training {name}...")
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else None

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else "N/A"

    results.append((name, accuracy, auc))
    print(f"✅ {name} - Accuracy: {accuracy:.4f}, AUC: {auc if auc == 'N/A' else f'{auc:.4f}'}")

print("\n=== 📋 Classifier Performance Summary ===")
for name, acc, auc in results:
    print(f"{name:20s} | Accuracy: {acc:.4f} | AUC: {auc if auc == 'N/A' else f'{auc:.4f}'}")
