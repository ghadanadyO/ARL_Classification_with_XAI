# combine CNN+Trad. +attention fusion 
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, Layer, Multiply, Concatenate, Reshape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pydicom
import cv2
import time

# ===== Load CSV and Excel Feature File =====
labeled_data_dir = 'Dataset 790'
labeled_csv_path = 'Dataset 790.csv'
# glcm_excel_path = '128_all_features_Normalization120c.xlsx'  # Your Excel file path
glcm_excel_path = '128_all_features_Normalization_Path2.xlsx'  # Your Excel file path

# Read both CSV and Excel
labeled_data = pd.read_csv(labeled_csv_path)
glcm_features_df = pd.read_excel(glcm_excel_path)

# Debug: Show column names to verify structure
print("Labeled CSV columns:", labeled_data.columns.tolist())
print("GLCM Excel columns:", glcm_features_df.columns.tolist())

# Rename the label column in the Excel file to avoid conflict during merge
glcm_features_df = glcm_features_df.rename(columns={'label': 'GLCM_Label'})

# Merge on Image_Path
combined_df = pd.merge(labeled_data, glcm_features_df, on='Image_Path')

# Now we safely access the correct label column from the CSV
label_col = 'label'
if label_col not in combined_df.columns:
    raise ValueError(f"Label column '{label_col}' not found in combined_df")

# Get paths and labels
image_paths = [os.path.join(labeled_data_dir, path) for path in combined_df['Image_Path']]
image_labels = combined_df[label_col].values


# Extract and normalize GLCM features
glcm_features = combined_df.drop(columns=[ 'Image_Path', 'label']).values
scaler = StandardScaler()
glcm_features = scaler.fit_transform(glcm_features)

print("Labeled Data Columns:", labeled_data.columns)
print("GLCM Feature Columns:", glcm_features_df.columns)
# ===== Load and preprocess DICOM images =====
labeled_images = []

for root, _, files in os.walk(labeled_data_dir):
    for file in files:
        if file.endswith('.dcm'):
            dicom_path = os.path.join(root, file)
            image = pydicom.dcmread(dicom_path)
            ct_image = image.pixel_array
            ct_image_resized = cv2.resize(ct_image, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
            ct_image_resized = np.expand_dims(ct_image_resized, axis=-1)  # Add channel dimension
            ct_image_normalized = tf.cast(ct_image_resized, tf.float32) / 255.0  # Normalize pixel values to [0, 1]
            labeled_images.append(ct_image_normalized)

X_labeled = np.array(labeled_images)
y_labeled = np.array(image_labels)
print ('X_labeled:', len(X_labeled))
print ('y_labeled:', len(y_labeled))

# ===== Attention Layers =====
class ChannelAttention(Layer):
    def __init__(self, reduction_ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_mlp = models.Sequential([
            Dense(channel // self.reduction_ratio, activation='relu'),
            Dense(channel)
        ])

    def call(self, inputs):
        avg_pool = GlobalAveragePooling2D()(inputs)
        max_pool = tf.reduce_max(inputs, axis=(1, 2), keepdims=False)
        avg_pool = Reshape((1, 1, -1))(avg_pool)
        max_pool = Reshape((1, 1, -1))(max_pool)
        mlp_avg = self.shared_mlp(avg_pool)
        mlp_max = self.shared_mlp(max_pool)
        scale = tf.nn.sigmoid(mlp_avg + mlp_max)
        return Multiply()([inputs, scale])

class SpatialAttention(Layer):
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.conv2d = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        conv = self.conv2d(concat)
        return inputs * conv

channel_attn = ChannelAttention()
spatial_attn = SpatialAttention()

def attention_fusion(feature1, feature2):
    fused = layers.Concatenate()([feature1, feature2])
    fused = channel_attn(fused)
    fused = spatial_attn(fused)
    return fused

# ===== Base CNN Model =====
def build_base_model(input_shape=(128, 128, 1)):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    return Model(inputs, x)

# ===== Model Architecture =====
input1 = Input(shape=(128, 128, 1))
input2 = Input(shape=(128, 128, 1))
glcm_input = Input(shape=(glcm_features.shape[1],))

model1 = build_base_model()
model2 = build_base_model()

feat1 = model1(input1)
feat2 = model2(input2)

fused = attention_fusion(feat1, feat2)
cnn_output = GlobalAveragePooling2D()(fused)

# Combine CNN + features
combined = Concatenate()([cnn_output, glcm_input])
x = Dense(128, activation='relu')(combined)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[input1, input2, glcm_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ===== Train/Test Split =====
X_train_img, X_test_img, X_train_glcm, X_test_glcm, y_train, y_test = train_test_split(
    X_labeled, glcm_features, y_labeled, test_size=0.15, random_state=42
)

# ===== Model Training =====
start = time.time()
history = model.fit(
    [X_train_img, X_train_img, X_train_glcm],
    y_train,
    validation_split=0.17,
    epochs=10,
    batch_size=64
)
print(f"Training Time: {time.time() - start:.2f} seconds")

# ===== Evaluation =====
test_preds = model.predict([X_test_img, X_test_img, X_test_glcm])
test_auc = roc_auc_score(y_test, test_preds)
test_bin = (test_preds > 0.5).astype(int)
print("Test Accuracy:", model.evaluate([X_test_img, X_test_img, X_test_glcm], y_test, verbose=0)[1])
print("Test Precision:", precision_score(y_test, test_bin))
print("Test Recall:", recall_score(y_test, test_bin))
print("Test F1:", f1_score(y_test, test_bin))
print("Test AUC:", test_auc)

# ===== Confusion Matrix =====
conf_matrix = confusion_matrix(y_test, test_bin)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Cancer', 'Cancer'],
            yticklabels=['Non-Cancer', 'Cancer'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# ===== Training Curves =====
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
