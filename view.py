import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

if os.path.exists('augmented_train_ds'):
    train_ds = tf.data.experimental.load('augmented_train_ds')
    for sample in train_ds.take(1):  # Nimm nur das erste Element
        for image in sample[0]:
            plt.imshow(image.numpy())  # Konvertiere Tensor in ein NumPy-Array
            plt.show()