import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from datasets import load_dataset
import random as ra
import tensorflow_addons as tfa

ds = load_dataset("garythung/trashnet")

"""def preprocess_data(ds):
    def preprocess_image(example):
        image = example['image']
        label = example['label']
        image = tf.image.resize(tf.convert_to_tensor(image), (150, 150))
        image = image / 255.0
        return image, label

    ds = ds.map(preprocess_image, batched=True)
    return ds

preprocess_data(ds)"""


# Display the first image and its label
first_image = ds['train'][0]['image']
first_label = ds['train'][0]['label']

first_image = tf.image.resize(tf.convert_to_tensor(first_image), (150, 150))
first_image = first_image / 255.0

def augment_image(image):
    """Wendet zufällige Transformationen auf ein Bild an."""
    # Zufälliges Spiegeln
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Zufällige 90 grad Rotation 
    image = tf.image.rot90(image, k=ra.randint(0, 3))

    # Manuelle Änderung der Helligkeit
    image = tf.image.adjust_brightness(image, delta=0.5)

    return image

first_image = augment_image(first_image)

print(f"Label: {first_label}")
plt.imshow(first_image)
plt.show()