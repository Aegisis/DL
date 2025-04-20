import tensorflow as tf
import matplotlib.pyplot as plt
import os
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

url = ("https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip")
cached_file_path = tf.keras.utils.get_file("cats_and_dogs_filtered.zip", url, cache_dir=os.getcwd())
with zipfile.ZipFile(cached_file_path, "r") as zip_ref:
    zip_ref.extractall()

train_dir = os.path.join(os.getcwd(), "cats_and_dogs_filtered", "train")
validation_dir = os.path.join(os.getcwd(), "cats_and_dogs_filtered", "validation")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode="binary"
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode="binary"
)

conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
conv_base.trainable = False

model = tf.keras.models.Sequential([
    conv_base,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=2e-5),
    metrics=["accuracy"]
)

history = model.fit(
    train_generator,
    steps_per_epoch=5,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=50
)

x, y_true = next(validation_generator)
y_pred = model.predict(x)
class_names = ['cat', 'dog']

for i in range(len(x)):
    plt.imshow(x[i])
    plt.title(f'Predicted: {class_names[int(round(y_pred[i][0]))]}, True: {class_names[int(y_true[i])]}')
    plt.show()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Training")
plt.plot(history.history['val_accuracy'], label="Validation")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Training")
plt.plot(history.history['val_loss'], label="Validation")
plt.title("Loss")
plt.legend()

plt.show()