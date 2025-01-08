import tensorflow as tf
#from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

if os.path.exists('train_ds'):
    train_ds = tf.data.experimental.load('train_ds')
    '''
    imgplot = plt.imshow(train_ds['train'][1]['image'])
    plt.show()'''
    for sample in train_ds.take(1):  # Nimm nur das erste Element
        for image in sample[0]:
            #image = sample[0][19]  # Zugriff auf das 'image'-Feld
            plt.imshow(image.numpy())  # Konvertiere Tensor in ein NumPy-Array
            plt.show()