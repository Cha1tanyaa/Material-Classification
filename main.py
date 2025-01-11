import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from datasets import load_dataset
import os
import matplotlib.pyplot as plt

# Load the dataset
ds = load_dataset("garythung/trashnet")

# Shuffle the dataset
ds = ds['train'].shuffle(seed=42)

# Use only 1/2 of the dataset
# ds = ds.shard(num_shards=1, index=0)

# Preprocess the dataset
def preprocess_data(example):
    image = tf.image.resize(example['image'], (100, 100)) / 255.0 # downscaling and normalization
    label = tf.cast(example['label'], tf.int32)
    return {'image': image, 'label': label}

# Split the dataset into train, validation, and test sets
train_test_split = ds.train_test_split(test_size=0.2)
train_val_split = train_test_split['train'].train_test_split(test_size=0.25)

# preprocess data and put it into batches
def to_tf_dataset(ds):
    ds = ds.map(preprocess_data)
    ds = ds.to_tf_dataset(columns=['image'], label_cols=['label'], batch_size=20, shuffle=True)
    return ds

# Check if preprocessed datasets exist
if not os.path.exists('train_ds') or not os.path.exists('validation_ds') or not os.path.exists('augmented_train_ds'):
    #preprocess data
    train_ds = to_tf_dataset(train_val_split['train'])
    validation_ds = to_tf_dataset(train_val_split['test'])
    test_ds = to_tf_dataset(train_test_split['test'])
    # Save the datasets
    tf.data.experimental.save(train_ds, 'train_ds')
    tf.data.experimental.save(validation_ds, 'validation_ds')
    tf.data.experimental.save(test_ds, 'test_ds')
else:
    # Load the datasets
    augmented_train_ds = tf.data.experimental.load('augmented_train_ds')
    validation_ds = tf.data.experimental.load('validation_ds')
    test_ds = tf.data.experimental.load('test_ds')

# define the model
model = Sequential([
    Input(shape=(100, 100, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.1),
    Dense(6, activation='softmax')  
])

# Compile the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the model
history = model.fit(
    augmented_train_ds,
    validation_data=validation_ds,
    epochs=15,
    callbacks=[early_stopping]
)

# get training history
training_loss = history.history['loss']
validation_loss = history.history['val_loss']
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

# plot of loss
plt.subplot(1, 2, 1)
plt.plot(training_loss)
plt.plot(validation_loss)
plt.title("loss curve")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["training loss", "validation loss"])

# plot of accuracy
plt.subplot(1, 2, 2)
plt.plot(training_accuracy)
plt.plot(validation_accuracy)
plt.title("accuracy curve")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(["training accuracy", "validation accuracy"])

plt.tight_layout()
plt.show()
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_ds)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')

# Save the model
model.save('trash_classifier_model.keras')