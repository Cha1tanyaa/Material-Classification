import tensorflow as tf

# Pfad zum gespeicherten Datensatz
DATASET_PATH = 'augmented_train_ds'

# Überprüfen, ob der Datensatz existiert
if not tf.io.gfile.exists(DATASET_PATH):
    raise FileNotFoundError(f"Datensatz nicht gefunden: {DATASET_PATH}")

# Datensatz laden
dataset = tf.data.experimental.load(DATASET_PATH)
print(f"Datensatz erfolgreich geladen: {DATASET_PATH}")

# Anzahl der Bilder zählen
image_count = [sample[0].shape[0] for sample in dataset]
print(f"Der Datensatz enthält {image_count} Bilder.")