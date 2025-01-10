import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from datasets import load_dataset

ds = load_dataset("garythung/trashnet")

def preprocess_data(ds):
    def preprocess_image(example):
        image = example['image']
        label = example['label']
        image = tf.image.resize(tf.convert_to_tensor(image), (150, 150))
        image = image / 255.0
        return image, label

    ds = ds.map(preprocess_image)
    return ds

preprocess_data(ds)


# Display the first image and its label
first_image = ds['train'][1000]['image']
first_label = ds['train'][1000]['label']

print(f"Label: {first_label}")
plt.imshow(first_image)
plt.show()