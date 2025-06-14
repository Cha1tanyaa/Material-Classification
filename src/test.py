import tensorflow as tf
import matplotlib.pyplot as plt
from datasets import load_dataset
import random as ra
import tensorflow_addons as tfa
import numpy as np

# --- Image Loading and Preprocessing ---
def load_and_preprocess_sample_image(dataset_name="garythung/trashnet", index=0, target_size=(150, 150)):
    """Loads a single image from the dataset, resizes, and normalizes it."""

    ds = load_dataset(dataset_name)
    print("Dataset loaded.")

    image_data = ds['train'][index]['image']
    label = ds['train'][index]['label']
    print(f"Original image type: {type(image_data)}, Label: {label}")

    # Convert to tensor if it's a PIL Image
    if not isinstance(image_data, tf.Tensor):
        image = tf.keras.preprocessing.image.img_to_array(image_data)
    else:
        image = image_data
    
    image = tf.image.resize(image, target_size)
    image = image / 255.0
    print(f"Preprocessed image shape: {image.shape}")
    return image, label

# --- Augmentation Functions (can be specific for testing) ---
def test_augment_image(image):
    """Applies specific transformations for testing purposes."""

    k = ra.randint(0, 3) 
    angle_to_rotate = ra.uniform(-np.pi/4, np.pi/4)
    brightness_delta = ra.uniform(-0.3, 0.3)
    augmented_image = tf.image.random_flip_left_right(image)
    augmented_image = tf.image.random_flip_up_down(augmented_image)
    augmented_image = tf.image.rot90(augmented_image, k=k)
    augmented_image = tfa.image.rotate(augmented_image, angles=angle_to_rotate, interpolation='BILINEAR')
    augmented_image = tf.image.random_brightness(augmented_image, max_delta=abs(brightness_delta))
    augmented_image = tf.clip_by_value(augmented_image, 0.0, 1.0)
    return augmented_image

# --- Display Function ---
def display_image(image, title="Sample Image"):
    """Displays a single image."""
    plt.figure()
    plt.imshow(image.numpy() if isinstance(image, tf.Tensor) else image)
    plt.title(title)
    plt.axis('off')
    plt.show()

# --- Main Test Script ---
if __name__ == "__main__":

    # 1. Load and preprocess a sample image
    sample_image, sample_label = load_and_preprocess_sample_image(index=5, target_size=(100,100))
    display_image(sample_image, title=f"Original Preprocessed Image\nLabel: {sample_label}")

    # 2. Augment the sample image
    augmented_sample_image = test_augment_image(tf.identity(sample_image))
    display_image(augmented_sample_image, title=f"Augmented Image\nLabel: {sample_label}")
    print("Test script executed successfully.")