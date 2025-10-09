import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.layers import Conv2D
from matplotlib import cm
from tqdm import tqdm

# ------------------- Grad-CAM Function -------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Compute gradients
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-6)
    return heatmap.numpy()


# ------------------- Display Function -------------------
def display_gradcam(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = np.array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img * 255
    return np.uint8(superimposed_img)


# ------------------- Grad-CAM Visualization -------------------
gradcam_output_dir = "GradCAM_All_Images"
os.makedirs(gradcam_output_dir, exist_ok=True)

# 🔹 Replace these lines:
# X_test_labeled, y_test_labeled → use all available dataset
# Example: combining labeled and unlabeled data arrays
X_all = np.concatenate([X_labeled, X_unlabeled], axis=0)
y_all = np.concatenate([y_labeled, y_unlabeled], axis=0) if 'y_unlabeled' in globals() else y_labeled

n_images = len(X_all)
print(f"🧠 Generating Grad-CAM visualizations for {n_images} total images...")

# Find the last Conv2D layer automatically
last_conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, Conv2D):
        last_conv_layer_name = layer.name
        break

if last_conv_layer_name is None:
    raise ValueError("❌ No Conv2D layer found in the model for Grad-CAM computation.")

# ------------------- Loop over all dataset images -------------------
for idx in tqdm(range(n_images), desc="Generating Grad-CAM"):
    image_to_explain = X_all[idx:idx + 1]
    true_label = y_all[idx] if y_all is not None else "N/A"
    prediction_score = model.predict(image_to_explain, verbose=0)[0][0]

    # Generate heatmap
    heatmap = make_gradcam_heatmap(image_to_explain, model, last_conv_layer_name)

    # Prepare the image
    img = image_to_explain[0]
    if img.shape[-1] == 1:
        img_rgb = np.repeat(img, 3, axis=-1)
    else:
        img_rgb = img
    if img_rgb.max() <= 1.0:
        img_rgb = img_rgb * 255
    img_rgb = img_rgb.astype(np.uint8)

    # Resize and overlay heatmap
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap_color, 0.4, img_rgb, 0.6, 0)

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(22, 14))
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Image', fontsize=20)
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=20)
    axes[2].imshow(superimposed_img[..., ::-1])
    axes[2].set_title('Superimposed', fontsize=20)
    for ax in axes:
        ax.axis('off')

    plt.suptitle(f"True: {true_label} | Pred: {round(prediction_score, 2)}", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save each Grad-CAM output
    save_path = os.path.join(gradcam_output_dir, f'gradcam_{idx+1}.png')
    plt.savefig(save_path)
    plt.close()

print("\n✅ Grad-CAM visualization for all dataset images completed successfully!")
