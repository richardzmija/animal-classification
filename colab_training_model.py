"""
Highly optimized TensorFlow code together with special features
of Google Colab for copying the dataset from Google Drive to
the local environment for dramatically faster execution.
"""

import os
import tensorflow as tf
import time
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import mixed_precision
from google.colab import drive

# Google drive handling
drive.mount("/content/drive")

# copy dataset to local storage
# UNCOMMENT IN COLAB
# !cp -r /content/drive/MyDrive/data/animal-classification /content/dataset

# File paths
folder_path = "/content/dataset"
train_path = os.path.join(folder_path, "train")
validate_path = os.path.join(folder_path, "validate")
test_path = os.path.join(folder_path, "test")

batch_size = 256

# Define a rule for data augmentation
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
    layers.experimental.preprocessing.RandomContrast(0.1),
    layers.experimental.preprocessing.RandomTranslation(height_factor=0.1,
                                                        width_factor=0.1)
])

train_dataset = image_dataset_from_directory(
    train_path,
    image_size=(128, 128),
    batch_size=batch_size,
    label_mode='categorical'
)

validation_dataset = image_dataset_from_directory(
    validate_path,
    image_size=(128, 128),
    batch_size=batch_size,
    label_mode='categorical'
)

test_dataset = image_dataset_from_directory(
    test_path,
    image_size=(128, 128),
    batch_size=batch_size,
    label_mode='categorical'    
)

# Apply data augmentation to the training dataset
train_dataset = train_dataset.map(lambda x, y:
                                  (data_augmentation(x, training=True), y))

# We can prefetch the data
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# Allow the network to use half-precision floating-point numbers
# for faster computations.
mixed_precision.set_global_policy("mixed_float16")

# Define the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(4, activation='softmax')
])

# Specify Adam and augment it as mixed precision optimizer
optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Train the model
start_time = time.time()

with tf.device('/GPU:0'):
    history = model.fit(
        train_dataset,
        epochs=50,
        validation_data=validation_dataset
    )

end_time = time.time()

# Display the post-training information
print(f'Training time: {end_time - start_time:.2f} seconds')

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test accuracy: {test_acc}')

model.save("trained_model.h5")
