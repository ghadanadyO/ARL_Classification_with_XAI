
import numpy as np
import pandas as pd
import cv2
import pydicom
import os
import gc
import psutil
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Conv2D, MaxPooling2D, GlobalAveragePooling2D

# Step 1: Load Traditional Features
csv_file = "128_all_features_Normalization2.csv"
df = pd.read_csv(csv_file)
traditional_features = df.drop(columns=['label']).values
labels = df['label'].values

# Step 2: Load and Preprocess CT Images
labeled_data_dir = 'Dataset 790'
labeled_csv_path = 'Dataset 790.csv'
labeled_images = []
image_labels = []
labeled_data = pd.read_csv(labeled_csv_path)
labeled_image_paths = [os.path.join(labeled_data_dir, row['Image_Path']) for _, row in labeled_data.iterrows()]
image_labels = labeled_data['label'].values

for root, _, files in os.walk(labeled_data_dir):
    for file in files:
        if file.endswith('.dcm'):
            dicom_path = os.path.join(root, file)
            image = pydicom.dcmread(dicom_path)
            ct_image = image.pixel_array
            ct_image_resized = cv2.resize(ct_image, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
            ct_image_resized = np.expand_dims(ct_image_resized, axis=-1)
            img = tf.cast(ct_image_resized, tf.float32) / 255.0  # Normalize pixel values to [0, 1]
            labeled_images.append(img)  # Flatten the image

X_labeled = np.array(labeled_images)
y_labeled = np.array(labels)


print('X_labeled:', len(X_labeled))
print('y_labeled:', len(y_labeled))

# Step 3: Preprocess Traditional Features
scaler = StandardScaler()
traditional_features_scaled = scaler.fit_transform(traditional_features)

# Step 4: Build Simple CNN Model
def build_simple_cnn_model(input_shape=(128, 128, 1)):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = GlobalAveragePooling2D()(x)
    model = Model(inputs, x)
    return model

cnn_base = build_simple_cnn_model()
cnn_base.summary()

# Step 5: Combine Deep Features with Traditional Features
image_input = Input(shape=(128, 128, 1), name='image_input')
deep_features = cnn_base(image_input)

traditional_input = Input(shape=(traditional_features.shape[1],), name='traditional_input')
combined = Concatenate()([deep_features, traditional_input])

x = Dense(128, activation='relu')(combined)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

combined_model = models.Model(inputs=[image_input, traditional_input], outputs=output)

# Step 6: Compile the Model
combined_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 7: Train-Test Split
X_train_img, X_test_img, X_train_trad, X_test_trad, y_train, y_test = train_test_split(
    X_labeled, traditional_features_scaled, y_labeled, test_size=0.15, random_state=42)
print('X_train_img:', len(X_train_img))
print('X_test_img:', len(X_test_img))
print('X_train_trad:', len(X_train_trad))
print('X_test_trad:', len(X_test_trad))
print('y_train:', len(y_train))
print('y_test:', len(y_test))

import time
Training_start_time = time.time()
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = combined_model.fit([X_train_img, X_train_trad], y_train, epochs=60, batch_size=128,
                   validation_split=0.1765, callbacks=[early_stopping])
training_time = time.time() - Training_start_time

testing_start_time = time.time()
y_pred = (combined_model.predict([X_test_img, X_test_trad]) > 0.5).astype("int32")
testing_time = time.time() - testing_start_time

Test_accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']
final_train_accuracy = training_accuracy[-1]
final_validation_accuracy = validation_accuracy[-1]
print(f"Final Validation Accuracy: {final_validation_accuracy}")
print(f"Final Training Accuracy: {final_train_accuracy}")
print("Test Time:", testing_time)
print("Train Time:", training_time)
print(f"Accuracy Testing: {Test_accuracy}")
print(f"Precision testing: {precision}")
print(f"Recall testing : {recall}")
print(f"F1-Score testing: {f1}")
print(f"AUC Testing: {auc}")

training_loss = history.history['loss']
validation_loss = history.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(training_accuracy, label='Training Accuracy')
plt.plot(validation_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Cancer", "Cancer"])
plt.figure(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()
