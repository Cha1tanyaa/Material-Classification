import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from datasets import load_dataset

# Load the dataset
ds = load_dataset("garythung/trashnet")

# Use only 1/5th of the dataset
ds = ds['train'].shard(num_shards=100, index=0)

# Preprocess the dataset
def preprocess_data(example):
    image = tf.image.resize(example['image'], (150, 150)) / 255.0
    label = tf.cast(example['label'], tf.int32)
    return {'image': image, 'label': label}

# Split the dataset into train, validation, and test sets
train_test_split = ds.train_test_split(test_size=0.2)
train_val_split = train_test_split['train'].train_test_split(test_size=0.25)

def to_tf_dataset(ds):
    ds = ds.map(preprocess_data)
    ds = ds.to_tf_dataset(columns=['image'], label_cols=['label'], batch_size=20, shuffle=True)
    return ds

train_ds = to_tf_dataset(train_val_split['train'])
validation_ds = to_tf_dataset(train_val_split['test'])
test_ds = to_tf_dataset(train_test_split['test'])

# Build the model
model = Sequential([
    Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the model
history = model.fit(
    train_ds,
    epochs=15,
    validation_data=validation_ds
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_ds)
print(f'Test accuracy: {test_acc}')

# Save the model
model.save('trash_classifier_model.h5')
