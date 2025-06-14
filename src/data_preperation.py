import tensorflow as tf
import tensorflow_addons as tfa
from datasets import load_dataset
import os
import numpy as np
import random as ra

# --- Preprocessing Functions ---
def preprocess_data(example):
    """Resizes and normalizes an image, and casts the label to int32."""
    image = tf.image.resize(example['image'], (100, 100)) / 255.0
    label = tf.cast(example['label'], tf.int32)
    return {'image': image, 'label': label}

def to_tf_dataset(dataset_split, batch_size=20, shuffle=True):
    """Converts a Hugging Face dataset split to a TensorFlow dataset."""
    dataset_split = dataset_split.map(preprocess_data)
    dataset_split = dataset_split.to_tf_dataset(
        columns=['image'],
        label_cols=['label'],
        batch_size=batch_size,
        shuffle=shuffle
    )
    return dataset_split

# --- Augmentation Functions ---
def augment_image(image):
    """Applies random transformations to an image."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tfa.image.rotate(image, angles=ra.uniform(-np.pi, np.pi))
    image = tf.image.random_brightness(image, max_delta=0.2)
    return image

def augment_dataset(dataset, augment_count=4):
    """Adds augmented images to each batch in the dataset."""
    def augment_batch(images, labels):
        augmented_images_list = [images]
        augmented_labels_list = [labels]
        
        for _ in range(augment_count):
            aug_images = tf.map_fn(augment_image, images)
            augmented_images_list.append(aug_images)
            augmented_labels_list.append(labels)
        
        augmented_images = tf.concat(augmented_images_list, axis=0)
        augmented_labels = tf.concat(augmented_labels_list, axis=0)
        return augmented_images, augmented_labels

    return dataset.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)

# --- Main Data Preparation Script ---
if __name__ == "__main__":

    # Load the raw dataset
    ds_raw = load_dataset("garythung/trashnet")
    print("Raw dataset loaded.")

    # Shuffle the dataset
    ds_shuffled = ds_raw['train'].shuffle(seed=42)
    print("Dataset shuffled.")

    # Split the dataset: 60% train 20% validation, 20% test 
    train_val_test_split = ds_shuffled.train_test_split(test_size=0.2, seed=42)
    train_val_split = train_val_test_split['train'].train_test_split(test_size=0.25, seed=42) 
    raw_train_ds = train_val_split['train']
    raw_validation_ds = train_val_split['test']
    raw_test_ds = train_val_test_split['test']
    print(f"Dataset split: {len(raw_train_ds)} train, {len(raw_validation_ds)} validation, {len(raw_test_ds)} test samples.")

    # Process and save initial datasets
    train_ds = to_tf_dataset(raw_train_ds, shuffle=True)
    validation_ds = to_tf_dataset(raw_validation_ds, shuffle=False)
    test_ds = to_tf_dataset(raw_test_ds, shuffle=False)

    tf.data.experimental.save(train_ds, 'data/train_ds')
    tf.data.experimental.save(validation_ds, 'data/validation_ds')
    tf.data.experimental.save(test_ds, 'data/test_ds')
    
    augmented_train_ds = augment_dataset(train_ds, augment_count=4)

    # Save the augmented training dataset
    tf.data.experimental.save(augmented_train_ds, 'data/augmented_train_ds')
    print(f"Augmented training dataset saved as: 'augmented_train_ds'")