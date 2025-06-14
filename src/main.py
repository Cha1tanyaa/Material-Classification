import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
import matplotlib.pyplot as plt

MODEL_SAVE_PATH = 'models/material_classifier_model.keras'

# --- Load preprocessed datasets ---
AUGMENTED_TRAIN_DS_PATH = 'data/augmented_train_ds'
VALIDATION_DS_PATH = 'data/validation_ds'
TEST_DS_PATH = 'data/test_ds'

augmented_train_ds = tf.data.experimental.load(AUGMENTED_TRAIN_DS_PATH)
validation_ds = tf.data.experimental.load(VALIDATION_DS_PATH)
test_ds = tf.data.experimental.load(TEST_DS_PATH)

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
    Dropout(0.2),
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
print("Model training finished.")

# get training history
training_loss = history.history['loss']
validation_loss = history.history['val_loss']
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

# plot of loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# plot of accuracy
plt.subplot(1, 2, 2)
plt.plot(training_accuracy, label='Training Accuracy')
plt.plot(validation_accuracy, label='Validation Accuracy')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_ds)
print(f'Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}')

# Save the model
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {os.path.abspath(MODEL_SAVE_PATH)}")