import tensorflow as tf
import tensorflow_addons as tfa
import os
import numpy as np
import random as ra

# Funktion für die Datenaugmentation
def augment_image(image):
    """Wendet zufällige Transformationen auf ein Bild an."""
    # Zufälliges Spiegeln
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Zufällige Rotation 
    image = tfa.image.rotate(image, angles=ra.uniform(-np.pi, np.pi))

    # Zufällige Änderung der Helligkeit
    image = tf.image.random_brightness(image, max_delta=0.2)

    return image

# Funktion zur Augmentation des Datasets
def augment_dataset(dataset, augment_count=2):
    #Fügt augmentierte Bilder zu jedem Batch hinzu.
    def augment_batch(images, labels):
        # Originalbilder beibehalten
        augmented_images = [images]
        augmented_labels = [labels]
        
        # Augmentierte Bilder hinzufügen
        for _ in range(augment_count):
            aug_images = tf.map_fn(augment_image, images)
            augmented_images.append(aug_images)
            augmented_labels.append(labels)
        
        # Alle Bilder und Labels zusammenfügen
        augmented_images = tf.concat(augmented_images, axis=0)
        augmented_labels = tf.concat(augmented_labels, axis=0)
        return augmented_images, augmented_labels

    # Dataset transformieren
    return dataset.map(augment_batch)

if not os.path.exists('train_ds') or not os.path.exists('validation_ds') or not os.path.exists('test_ds'):
    raise FileNotFoundError("Dataset nicht gefunden!")

# Datensatz laden
train_ds = tf.data.experimental.load('train_ds')
validation_ds = tf.data.experimental.load('validation_ds')

# Augmentieren des Trainingsdatensatzes
augmented_ds = augment_dataset(train_ds, augment_count=4)

# Speichern des augmentierten Datensatzes
tf.data.experimental.save(augmented_ds, 'augmented_train_ds')
print(f"Augmentierter Datensatz gespeichert unter: {'augmented_train_ds'}")