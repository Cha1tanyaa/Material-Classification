import tensorflow as tf
import matplotlib.pyplot as plt
import os

AUGMENTED_TRAIN_DS_PATH = '../data/augmented_train_ds'

def view_dataset_samples(dataset_path, num_batches_to_show=1, num_images_per_batch_to_show=5):
    """
    Loads a TensorFlow dataset and displays a few sample images.
    """

    dataset = tf.data.experimental.load(dataset_path)
    print("Dataset loaded.")

    for i, batch in enumerate(dataset.take(num_batches_to_show)):
        images, labels = batch
        print(f"Batch {i+1}:")
        for j in range(min(num_images_per_batch_to_show, images.shape[0])):
            plt.figure()
            plt.imshow(images[j].numpy())
            plt.title(f"Sample Image {j+1} from Batch {i+1}\nLabel: {labels[j].numpy()}")
            plt.axis('off')
            plt.show()
        if i + 1 >= num_batches_to_show:
            break

if __name__ == "__main__":
    dataset_file_path = AUGMENTED_TRAIN_DS_PATH 
    script_dir = os.path.dirname(__file__)
    dataset_file_path = os.path.normpath(os.path.join(script_dir, AUGMENTED_TRAIN_DS_PATH))

    view_dataset_samples(dataset_file_path, num_batches_to_show=1, num_images_per_batch_to_show=3)
    print("Finished displaying dataset samples.")