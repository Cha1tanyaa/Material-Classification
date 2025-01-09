'''import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

if os.path.exists('train_ds'):
    train_ds = tf.data.experimental.load('train_ds')
    for sample in train_ds.take(1):  # Nimm nur das erste Element
        for image in sample[0]:
            #image = sample[0][19]  # Zugriff auf das 'image'-Feld
            image2 = tf.image.random_flip_left_right(image)
            plt.imshow(image2.numpy())  # Konvertiere Tensor in ein NumPy-Array
            plt.show()'''

import tensorflow as tf
import tensorflow_addons as tfa
import os
import numpy as np

# Funktion für die Datenaugmentation
def augment_image(image):
    """Wendet zufällige Transformationen auf ein Bild an."""
    # Zufälliges Spiegeln
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Zufällige Rotation (um maximal 20 Grad)
    angle = np.random.uniform(-20, 20) * np.pi / 180
    image = tfa.image.rotate(image, angles=angle)

    # Zufällige Änderung der Helligkeit
    image = tf.image.random_brightness(image, max_delta=0.2)

    return image

# Pfad zum gespeicherten Datensatz
AUGMENTED_PATH = 'augmented_train_ds'

if not os.path.exists('train_ds') or not os.path.exists('validation_ds') or not os.path.exists('test_ds'):
    raise FileNotFoundError(f"Dataset nicht gefunden: {DATASET_PATH}")

# Datensatz laden
train_ds = tf.data.experimental.load('train_ds')
validation_ds = tf.data.experimental.load('validation_ds')

augmented_data = []
# Iteriere durch den Datensatz
count = 0
print("\n\nTyp: ", type(train_ds), "\n\n")
for sample in train_ds:
    augmented_data.append(sample)  # Originalbild beibehalten
    image = sample[0]
    label = sample[1]
    count += 1
    print(f"Augmentiere Batch: {count}")
    # Wende Datenaugmentation an, erstelle 2-4 neue Bilder
    for i in range(label.shape[0]):
        for _ in range(np.random.randint(2, 5)):
            augmented_sample = {
                'image': augment_image(image[i]),
                'label': label[i]
            }
            augmented_data.append(augmented_sample)
print("\n\nBearbeitung fertig\n\n")
# Augmentierte Daten als neues Dataset speichern
augmented_ds = tf.data.Dataset.from_generator(
    lambda: (data for data in augmented_data),
    output_signature={
        'image': tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
        'label': tf.TensorSpec(shape=(), dtype=tf.int32)
    }
)
print("\n\nDatensatz erstellt\n\n")
# Speichern des augmentierten Datensatzes
tf.data.experimental.save(augmented_ds, AUGMENTED_PATH)
print(f"Augmentierter Datensatz gespeichert unter: {AUGMENTED_PATH}")