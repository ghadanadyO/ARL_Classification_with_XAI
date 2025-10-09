import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Input, \
    LayerNormalization, MultiHeadAttention, Add
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
import pydicom
import cv2
import gc
import psutil
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# from sklearn.metrics import roc_auc_score
import seaborn as sns  # For confusion matrix heatmap
import time
from tensorflow.keras.regularizers import l2
from keras.utils import to_categorical
from collections import defaultdict
import tensorflow as tf
import matplotlib.pyplot as plt  # Import matplotlib for plotting

def build_simplified_coatnet_model(input_shape=(128, 128, 1)):
    inputs = Input(shape=input_shape)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)

    for _ in range(2):  # Simplified blocks
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Dropout(0.2)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, x)

    model = Model(inputs=inputs, outputs=output)

    # Step 6: Compile the Model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Initialize variables to track training history
initial_training_history = None
active_learning_training_histories = []

# Create lists to store training and validation accuracy and loss
training_accuracy = []
validation_accuracy = []
training_loss = []
validation_loss = []
initial_training_time = 0
active_learning_training_times = []
# Initialize variables to track training and validation accuracies
initial_training_accuracy = 0
active_learning_training_accuracies = []
# Initialize variables to track AUC scores
initial_auc = 0
active_learning_auc = []



# Set paths to your labeled and unlabeled image directories
# unlabeled_data_dir = 'D:/PHD/DSB3/stage1/stage1/used_All_Labeled22Person'
# unlabeled_csv_path = 'D:/PHD/DSB3/stage1/stage1/used_All_Labeled22Person.csv'
labeled_data_dir = 'labeled80'
unlabeled_data_dir = 'unlabeled710'
labeled_csv_path = 'labeled80.csv'
unlabeled_csv_path = 'unlabeled710.csv'
# Load labeled data from Excel files
labeled_images = []
labels = []

labeled_data = pd.read_csv(labeled_csv_path)
labeled_image_paths = [os.path.join(labeled_data_dir, row['Image_Path']) for _, row in labeled_data.iterrows()]
labels = labeled_data['label'].values
# unlabeled_csv_path = '/excel files csv/unlabeled2.csv'
unlabeled_data = pd.read_csv(unlabeled_csv_path)
unlabeled_image_paths = [os.path.join(unlabeled_data_dir, row['Image_Path']) for _, row in unlabeled_data.iterrows()]

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

# Make sure the number of samples matches the number of labels
# assert len(X_labeled) == len(y_labeled), "Number of samples and labels do not match"
print('X_labeled:', len(X_labeled))
print('y_labeled:', len(y_labeled))


unlabeled_images = []

for root, _, files in os.walk(unlabeled_data_dir):
    for file in files:
        if file.endswith('.dcm'):
            dicom_path = os.path.join(root, file)
            image = pydicom.dcmread(dicom_path)
            ct_image = image.pixel_array
            ct_image = image.pixel_array
            ct_image_resized = cv2.resize(ct_image, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
            ct_image_resized = np.expand_dims(ct_image_resized, axis=-1)
            img = tf.cast(ct_image_resized, tf.float32) / 255.0  # Normalize pixel values to [0, 1]
            unlabeled_images.append(img)  # Flatten the image

X_unlabeled = np.array(unlabeled_images)
print('X_unlabeled:', len(X_unlabeled))
# Initialize variables to track training and validation times
initial_training_time = 0
initial_validation_time = 0  # Add this line
active_learning_training_times = []
active_learning_validation_times = []  # Add this line
X_train_labeled, X_temp, y_train_labeled, y_temp = train_test_split(
    X_labeled, y_labeled, test_size=0.3, random_state=42)  # 30% of data will be used for validation and testing

# Split the temp set into validation and test sets
X_val_labeled, X_test_labeled, y_val_labeled, y_test_labeled = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)  # 50% of temp data will be used for validation and testing

# Check the shapes to ensure the split is correct
print("Training set size:", X_train_labeled.shape)
print("Validation set size:", X_val_labeled.shape)
print("Test set size:", X_test_labeled.shape)


def dataset_to_numpy(dataset):
    features = []
    labels = []
    for x_batch, y_batch in dataset:
        features.append(x_batch.numpy())
        labels.append(y_batch.numpy())
    return np.concatenate(features), np.concatenate(labels)


# Convert data to TensorFlow dataset objects
def create_tf_dataset(X, y, batch_size, shuffle_buffer_size=1000):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


train_dataset = create_tf_dataset(X_train_labeled, y_train_labeled, batch_size=16)
val_dataset = create_tf_dataset(X_val_labeled, y_val_labeled, batch_size=16)

# Convert TensorFlow dataset objects to NumPy arrays
X_train_labeled, y_train_labeled = dataset_to_numpy(train_dataset)
X_val_labeled, y_val_labeled = dataset_to_numpy(val_dataset)

# ................................. Train initial model.............................................
model =  build_simplified_coatnet_model(input_shape=(128, 128, 1))
model.summary()
# Train initial model
initial_start_time = time.time()

# Train your model
history = model.fit(np.array(X_train_labeled), np.array(y_train_labeled), epochs=20, batch_size=64,
                    validation_data=(np.array(X_val_labeled), np.array(y_val_labeled)))

# Calculate the training time for the initial model
initial_training_time = time.time() - initial_start_time
# Calculate and display training accuracy for the initial model
initial_training_accuracy = model.evaluate(X_train_labeled, y_train_labeled, verbose=0)[1]
# Calculate and display validation time for the initial model
initial_start_time = time.time()
initial_validation_accuracy = model.evaluate(X_val_labeled, y_val_labeled, verbose=0)[1]
initial_validation_time = time.time() - initial_start_time
# Calculate and display AUC for the initial model
initial_predictions = model.predict(X_val_labeled)
initial_auc = roc_auc_score(y_val_labeled, initial_predictions)
print("Initial Model Training Time:", initial_training_time)
print("Initial Model Training Accuracy:", initial_training_accuracy)
print("Initial Model Validation Accuracy:", initial_validation_accuracy)
print("Initial Model Validation Time:", initial_validation_time)  # Add this line
print("Initial Model AUC:", initial_auc)

# Convert the model's predictions to binary labels (0 or 1) using a threshold (0.5 in this case)
threshold = 0.5
initial_predictions_binary = (initial_predictions > threshold).astype(int)
# Calculate and display precision for the initial model
initial_precision = precision_score(y_val_labeled, initial_predictions_binary)
# Calculate and display recall for the initial model
initial_recall = recall_score(y_val_labeled, initial_predictions_binary)
# Calculate and display F1-score for the initial model
initial_f1 = f1_score(y_val_labeled, initial_predictions_binary)
print("Initial Model Precision:", initial_precision)
print("Initial Model F1-Score:", initial_f1)
print("Initial Model Recall:", initial_recall)

# Get the training history
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

# Plot training and validation accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(training_accuracy, label='Training Accuracy')
plt.plot(validation_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()  # Display the plots

# .........................testing...................
# Evaluate the model on the test set
test_start_time = time.time()
test_accuracy = model.evaluate(X_test_labeled, y_test_labeled, verbose=0)[1]
test_time = time.time() - test_start_time
print("Test Accuracy:", test_accuracy)
print("Test Time:", test_time)

# Get predictions on the test set
test_predictions = model.predict(X_test_labeled)
test_auc = roc_auc_score(y_test_labeled, test_predictions)
print("Test AUC:", test_auc)

# Convert predictions to binary labels
test_predictions_binary = (test_predictions > threshold).astype(int)

# Calculate precision, recall, and F1-score
test_precision = precision_score(y_test_labeled, test_predictions_binary)
test_recall = recall_score(y_test_labeled, test_predictions_binary)
test_f1 = f1_score(y_test_labeled, test_predictions_binary)

print("Test Precision:", test_precision)
print("Test Recall:", test_recall)
print("Test F1-Score:", test_f1)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test_labeled, test_predictions_binary)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Cancer', 'Cancer'],
            yticklabels=['Non-Cancer', 'Cancer'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

#..................................start Active reinforcement learning.......................
print('X_labeled:', len(X_labeled))
# -------------------------- Initial Setup --------------------------

# Monitor initial GPU and RAM usage
print_gpu_memory()
print_ram_usage()

# Clear memory before starting
clear_memory()

# Enable mixed precision training if supported
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled.")
except Exception as e:
    print(f"Failed to set mixed precision: {e}")

# Clear memory after setting policies
clear_memory()
print_ram_usage()

print_ram_usage()
clear_memory()
clear_memory()
clear_memory()
clear_memory()
clear_memory()
print_ram_usage()
# ........................... initial variable..................................
active_learning_training_times = []
active_learning_validation_times = []
active_learning_testing_times = []
active_learning_training_accuracies = []
active_learning_auc = []
accuracy_scores = []

train_accuracy_scores = []
train_precision_scores = []
train_auc_score = []
train_recall_scores = []
train_f1_scores = []
train_roc_auc_scores = []
test_confusion_matrices = []
val_precision_scores = []
val_accuracy_scores = []
val_recall_scores = []
val_f1_scores = []
val_auc_scores = []
test_accuracy_scores = []
test_precision_scores = []
test_recall_scores = []
test_f1_scores = []
test_auc_score = []
# Active learning with Q-learning
n_queries = 120 # Number of queries (iterations)
# n_queries = 2
batch_size = 200  # Start with a batch size of 32
learning_rate = 0.1  # Learning rate for Q-learning
gamma = 0.95  # Discount factor for Q-learning
epsilon = 0.1  # Exploration rate
# Initialize Q-table
Q_table = defaultdict(lambda: np.zeros(2))  # Two possible actions: 0 and 1
# Example active learning loop
for query in range(n_queries):
    print(f"Active learning iteration {query + 1}/{n_queries}")
    # Select samples to label
    query_start_time = time.time()
    for batch in range(0, len(X_unlabeled), batch_size):
        current_batch = X_unlabeled[batch:batch + batch_size]
        # predicted_probs = model.predict(current_batch)
        predicted_probs = model.predict(current_batch, batch_size=8)
        # num_samples_to_predict = min(batch_size, len(X_unlabeled))  # Ensure batch_size is not larger than the number of unlabeled samples
        # current_batch = X_unlabeled[:num_samples_to_predict]
        # predicted_probs = model.predict(current_batch)
        predicted_probs = np.clip(predicted_probs, 1e-10, 1.0)  # Avoid log(0) issues
        entropies = -np.sum(predicted_probs * np.log(predicted_probs), axis=1)
        kth = min(batch_size, len(entropies))  # Calculate a valid kth value
        top_indices = np.argpartition(entropies, -kth)[-kth:]  # Use the valid kth value for partitioning
        selected_indices = top_indices
        selected_images = current_batch[selected_indices]
        selected_labels = predicted_probs[selected_indices]
        # query_time = time.time() - query_start_time
        # Simulate labeling the selected samples (replace this with your actual labeling process)
        simulated_labels = np.random.randint(0, 2, size=len(selected_images))  # Simulate random labels
        selected_labels_binary = (selected_labels > 0.5).astype(int)
        # Update Q-table
        # Update Q-table
        for i, idx in enumerate(selected_indices):
            state = tuple(selected_images[i].flatten())
            action = simulated_labels[i]
            reward = 1 if simulated_labels[i] == selected_labels_binary[i] else -1
            next_state = tuple(X_unlabeled[idx].flatten())
            Q_value = Q_table[state][action]
            next_max_q_value = np.max(Q_table[next_state])
            Q_table[state][action] = Q_value + learning_rate * (reward + gamma * next_max_q_value - Q_value)

    # Update labeled and unlabeled data

    # Update labeled and unlabeled data
    print('X_labeled:', len(X_labeled))
    print('Length of X_labeled:', len(X_labeled))
    print('Length of y_labeled:', len(y_labeled))
    # X_labeled = np.concatenate((X_labeled, selected_images.reshape((-1, 32, 32, 1))))
    X_labeled = np.concatenate((X_labeled, selected_images))
    discretized_labels = (selected_labels > 0.5).astype(int)
    y_labeled = np.append(y_labeled, discretized_labels)
    X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
    labeled_image_paths.extend([unlabeled_image_paths[i] for i in selected_indices])
    unlabeled_image_paths = np.delete(unlabeled_image_paths, selected_indices)
    query_time = time.time() - query_start_time
    # query_time = time.time() - query_start_time
    # active_learning_training_times.append(query_time)
    # active_learning_validation_times.append(query_time)
    # active_learning_testing_times.append(query_time)
    # active_learning_training_accuracies.append(initial_training_accuracy)
    # active_learning_auc.append(test_auc)
    # precision_scores.append(test_precision)
    # recall_scores.append(test_recall)
    # f1_scores.append(test_f1)
    # confusion_matrices.append(confusion_matrix(y_test_labeled, initial_test_predictions_binary))
    print(f'Query {query + 1}/{n_queries} - Time: {query_time:.4f}s')
    print('X_labeled:', len(X_labeled))
    print('X_unlabeled: ', len(X_unlabeled))

    # Retrain model on updated labeled data
    # Split the data into training and temp sets (temp set will be further split into validation and test sets)
    X_train_labeled, X_temp, y_train_labeled, y_temp = train_test_split(
        X_labeled, y_labeled, test_size=0.3, random_state=42)  # 30% of data will be used for validation and testing

    # Split the temp set into validation and test sets
    X_val_labeled, X_test_labeled, y_val_labeled, y_test_labeled = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)  # 50% of temp data will be used for validation and testing

    # train_dataset = create_tf_dataset(X_train_labeled, y_train_labeled, batch_size=16)
    # val_dataset = create_tf_dataset(X_val_labeled, y_val_labeled, batch_size=16)
    # test_dataset = create_tf_dataset(X_test_labeled, y_test_labeled, batch_size=16)
    train_dataset = create_tf_dataset(X_train_labeled, y_train_labeled, batch_size=16).cache().prefetch(
        buffer_size=tf.data.AUTOTUNE)
    val_dataset = create_tf_dataset(X_val_labeled, y_val_labeled, batch_size=16).cache().prefetch(
        buffer_size=tf.data.AUTOTUNE)
    test_dataset = create_tf_dataset(X_test_labeled, y_test_labeled, batch_size=16).cache().prefetch(
        buffer_size=tf.data.AUTOTUNE)

    # Convert TensorFlow dataset objects to NumPy arrays
    X_train_labeled, y_train_labeled = dataset_to_numpy(train_dataset)
    X_val_labeled, y_val_labeled = dataset_to_numpy(val_dataset)
    X_test_labeled, y_test_labeled = dataset_to_numpy(test_dataset)

    training_start_time = time.time()
    # model.fit(X_train_labeled, y_train_labeled, epochs=15, batch_size=16, validation_data=(val_dataset))
    # active_learning_history = model.fit(train_dataset, epochs=15, validation_data=val_dataset)
    # active_learning_history = model.fit(np.array(X_train_labeled), np.array(y_train_labeled), epochs=15, batch_size=16,
    #                    validation_data=(np.array(X_val_labeled), np.array(y_val_labeled)))
    active_learning_history = model.fit(np.array(X_train_labeled), np.array(y_train_labeled),
                                        epochs=15, batch_size=64,  # Use a smaller batch size like 8 or 16
                                        validation_data=(np.array(X_val_labeled), np.array(y_val_labeled))
                                        )

    # Evaluate model performance on training data
    # active_learning_train_predictions = model.predict(X_train_labeled)
    # active_learning_train_predictions_binary = (active_learning_train_predictions > 0.5).astype(int)
    train_accuracy = model.evaluate(X_train_labeled, y_train_labeled, verbose=0)[1]
    train_predictions = model.predict(X_train_labeled)
    train_auc = roc_auc_score(y_train_labeled, train_predictions)
    train_precision = precision_score(y_train_labeled, (train_predictions > 0.5).astype(int))
    train_recall = recall_score(y_train_labeled, (train_predictions > 0.5).astype(int))
    train_f1 = f1_score(y_train_labeled, (train_predictions > 0.5).astype(int))
    # train_accuracy = accuracy_score(y_train_labeled, train_predictions)  # Accuracy calculation
    # training_accuracy = model.evaluate(X_train_labeled, y_train_labeled, verbose=0)[1]
    # print("Training Accuracy:", training_accuracy)
    # Store performance metrics
    train_auc_score.append(train_auc)
    train_precision_scores.append(train_precision)
    train_recall_scores.append(train_recall)
    train_f1_scores.append(train_f1)
    train_accuracy_scores.append(train_accuracy)
    # train_confusion_matrices.append(val_confusion)
    # Print training metrics
    print(
        f'Query {query + 1}/{n_queries} - Training - AUC: {train_auc:.4f},Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1_Score : {train_f1:.4f}')
    training_time = time.time() - training_start_time
    active_learning_training_times.append(training_time)

    # Evaluate the model validation data
    val_start_time = time.time()
    val_accuracy = model.evaluate(X_val_labeled, y_val_labeled, verbose=0)[1]
    val_predictions = model.predict(X_val_labeled)
    val_auc = roc_auc_score(y_val_labeled, val_predictions)
    val_precision = precision_score(y_val_labeled, (val_predictions > 0.5).astype(int))
    val_recall = recall_score(y_val_labeled, (val_predictions > 0.5).astype(int))
    val_f1 = f1_score(y_val_labeled, (val_predictions > 0.5).astype(int))
    val_predictions_binary = (val_predictions > 0.5).astype(int)
    # val_confusion=confusion_matrix(y_val_labeled, val_predictions_binary)
    # Store performance metrics
    val_auc_scores.append(val_auc)
    val_precision_scores.append(val_precision)
    val_recall_scores.append(val_recall)
    val_f1_scores.append(val_f1)
    val_accuracy_scores.append(val_accuracy)
    # val_confusion_matrices.append(val_confusion)

    # Print validation metrics
    print(
        f'Query {query + 1}/{n_queries} - Validation - AUC: {val_auc:.4f}, Accuracy:{val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1_Score: {val_f1:.4f}')
    val_time = time.time() - val_start_time
    active_learning_validation_times.append(val_time)
    # # Evaluate model performance on testing data
    # active_learning_test_predictions = model.predict(X_test_labeled)
    # active_learning_test_predictions_binary = (active_learning_test_predictions > 0.5).astype(int)
    test_start_time = time.time()
    test_accuracy = model.evaluate(X_test_labeled, y_test_labeled, verbose=0)[1]
    test_predictions = model.predict(X_test_labeled)
    test_auc = roc_auc_score(y_test_labeled, test_predictions)
    test_precision = precision_score(y_test_labeled, (test_predictions > 0.5).astype(int))
    test_recall = recall_score(y_test_labeled, (test_predictions > 0.5).astype(int))
    test_f1 = f1_score(y_test_labeled, (test_predictions > 0.5).astype(int))
    test_confusion = confusion_matrix(y_test_labeled, (test_predictions > 0.5).astype(int))
    # test_accuracy = accuracy_score(y_test_labeled, active_learning_test_predictions_binary)  # Accuracy calculation
    # Store performance metrics
    test_auc_score.append(test_auc)
    test_precision_scores.append(test_precision)
    test_recall_scores.append(test_recall)
    test_f1_scores.append(test_f1)
    test_accuracy_scores.append(test_accuracy)
    test_confusion_matrices.append(test_confusion)

    # Print testing metrics
    print(
        f'Query {query + 1}/{n_queries} - Testing - AUC: {test_auc:.4f},Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1_Score: {test_f1:.4f}')
    test_time = time.time() - test_start_time
    active_learning_testing_times.append(test_time)
    print("After iteration:")
    

# Evaluate the final model on the test set after active learning
final_test_start_time = time.time()
final_test_accuracy = model.evaluate(X_test_labeled, y_test_labeled, verbose=0)[1]
final_test_predictions = model.predict(X_test_labeled)
final_test_predictions_binary = (final_test_predictions > 0.5).astype(int)
final_test_auc = roc_auc_score(y_test_labeled, final_test_predictions)
final_test_precision = precision_score(y_test_labeled, final_test_predictions_binary)
final_test_recall = recall_score(y_test_labeled, final_test_predictions_binary)
final_test_f1 = f1_score(y_test_labeled, final_test_predictions_binary)
final_test_time = time.time() - final_test_start_time

# Print final model performance on the testing set
print(f"Final Testing Accuracy: {final_test_accuracy:.4f}")
print(f"Final AUC: {final_test_auc:.4f}")
print(f"Final Precision: {final_test_precision:.4f}")
print(f"Final Recall: {final_test_recall:.4f}")
print(f"Final F1-Score: {final_test_f1:.4f}")
print(f"Final Testing Time: {final_test_time:.4f}s")

print('Length of y_test_labeled:', len(y_test_labeled))
print('Length of test_predictions_binary:', len(test_predictions_binary))
# Compute confusion matrix
conf_matrix = confusion_matrix(y_test_labeled, final_test_predictions_binary)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Cancer', 'Cancer'],
            yticklabels=['Non-Cancer', 'Cancer'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
# Initialize an empty list to store the paths of labeled images
unlabeled_image_paths_to_save = unlabeled_image_paths
# Save the paths of labeled images with high entropy to an Excel file
unlabeled_paths_df = pd.DataFrame({'Image_Path': unlabeled_image_paths_to_save})  # Use unlabeled_image_paths_to_save
# Path to the Excel file to save the labeled images
unlabeled_excel_path = 'unlabeled_data_ALReinforcement.xlsx'
unlabeled_paths_df.to_excel(unlabeled_excel_path, index=False)

# Print the number of labeled and unlabeled images
print('labeled_image_paths: ', len(labeled_image_paths))
print('Label:', len(y_labeled))
# Save labeled data and metrics
labeled_paths_and_labels = {'path': labeled_image_paths, 'Label': y_labeled}
labeled_df = pd.DataFrame(labeled_paths_and_labels)
labeled_df.to_excel('labeled_data_ALReinforcement.xlsx', index=False)

query_numbers = list(range(1, len(active_learning_training_times) + 1))
# Define a function to pad lists
def pad_list(lst, length, value=None):
    return lst + [value] * (length - len(lst))


# Get the desired length (length of query_numbers)
desired_length = len(query_numbers)

# Pad or truncate lists
active_learning_training_times = pad_list(active_learning_training_times, desired_length)
train_precision_scores = pad_list(train_precision_scores, desired_length)
train_recall_scores = pad_list(train_recall_scores, desired_length)
train_auc_score = pad_list(train_auc_score, desired_length)
train_f1_scores = pad_list(train_f1_scores, desired_length)
active_learning_validation_times = pad_list(active_learning_validation_times, desired_length)
val_auc_scores = pad_list(val_auc_scores, desired_length)
val_accuracy_scores = pad_list(val_accuracy_scores, desired_length)
val_precision_scores = pad_list(val_precision_scores, desired_length)
val_recall_scores = pad_list(val_recall_scores, desired_length)
val_f1_scores = pad_list(val_f1_scores, desired_length)
active_learning_testing_times = pad_list(active_learning_testing_times, desired_length)
test_accuracy_scores = pad_list(test_accuracy_scores, desired_length)
test_auc_score = pad_list(test_auc_score, desired_length)
test_precision_scores = pad_list(test_precision_scores, desired_length)
test_recall_scores = pad_list(test_recall_scores, desired_length)
test_f1_scores = pad_list(test_f1_scores, desired_length)

# Ensure lists are populated and not empty
metrics_data = {
    'Query number': query_numbers,
    'Training Time': active_learning_training_times,
    'Train Precision': train_precision_scores,
    'Train Recall': train_recall_scores,
    'Train AUC': train_auc_score,
    'Train F1 Score': train_f1_scores,
    'Validation Time': active_learning_validation_times,
    'Validation AUC': val_auc_scores,
    'Validation Accuracy': val_accuracy_scores,
    'Validation Precision': val_precision_scores,
    'Validation Recall': val_recall_scores,
    'Validation F1 Score': val_f1_scores,
    'Testing Time': active_learning_testing_times,
    'Test Accuracy': test_accuracy_scores,
    'Test AUC': test_auc_score,
    'Test Precision': test_precision_scores,
    'Test Recall': test_recall_scores,
    'Test F1 Score': test_f1_scores
}

# Convert to DataFrame
metrics_df = pd.DataFrame(metrics_data)

# Check if DataFrame is not empty before saving
if not metrics_df.empty:
    metrics_df.to_excel('metrics_AL_entropy.xlsx', index=False)
    print("Metrics successfully saved to Excel")

# Optionally test writing to CSV
# metrics_df.to_csv('D:/PHD/DSB3/stage1/stage1/metrics_AL_entropy_test.csv', index=False)
# metrics_df.to_excel('D:/PHD/DSB3/stage1/stage1/metrics_AL_entropy.xlsx', index=False)

# Plot training and validation accuracy and loss
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
# Save the final Q-table
np.save('Q_table.npy', dict(Q_table))
# Convert the Q_table dictionary to a list of tuples for DataFrame creation
Q_table_list = [(state, action, value) for state, actions in Q_table.items() for action, value in enumerate(actions)]
# Convert the Q_table list to a DataFrame
df_array = pd.DataFrame(Q_table_list, columns=['State', 'Action', 'Value'])
# Save the DataFrame to an Excel file
df_array.to_excel('Q_table.xlsx', index=False)

# Plot the final performance metrics
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(range(1, len(test_precision_scores) + 1), test_precision_scores, label='Precision')
plt.xlabel('Query')
plt.ylabel('Precision')
plt.legend()

plt.subplot(2, 2, 2)
# plt.plot(range(1, n_queries + 1), recall_scores, label='Recall')
plt.plot(range(1, len(test_recall_scores) + 1), test_recall_scores, label='Recall')
plt.xlabel('Query')
plt.ylabel('Recall')
plt.legend()

plt.subplot(2, 2, 3)
# plt.plot(range(1, n_queries + 1), f1_scores, label='F1-Score')
plt.plot(range(1, len(test_f1_scores) + 1), test_f1_scores, label='F1-score')
plt.xlabel('Query')
plt.ylabel('F1-Score')
plt.legend()

plt.subplot(2, 2, 4)
# plt.plot(range(1, n_queries + 1), active_learning_auc, label='AUC')
plt.plot(range(1, len(test_auc_score) + 1), test_auc_score, label='AUC')
plt.xlabel('Query')
plt.ylabel('AUC')
plt.legend()
plt.tight_layout()
plt.show()


