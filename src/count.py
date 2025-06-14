import tensorflow as tf
import os

AUGMENTED_TRAIN_DS_PATH = '../data/augmented_train_ds'

def count_images_in_dataset(dataset_path):
    """
    Loads a TensorFlow dataset and counts the total number of images.
    Assumes the dataset yields batches of (images, labels).
    """

    dataset = tf.data.experimental.load(dataset_path)
    print("Dataset loaded successfully.")

    total_images = 0
    num_batches = 0
    for batch in dataset:
        images_in_batch = batch[0].shape[0]
        total_images += images_in_batch
        num_batches += 1
    
    return total_images

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    dataset_file_path = os.path.normpath(os.path.join(script_dir, AUGMENTED_TRAIN_DS_PATH))
    image_count = count_images_in_dataset(dataset_file_path)

    print(f"The dataset '{AUGMENTED_TRAIN_DS_PATH}' contains approximately {image_count} images.")
    print("Finished counting images in the dataset.")