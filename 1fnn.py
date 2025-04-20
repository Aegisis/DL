import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelBinarizer

iris = load_iris()

X = iris.data
y = iris.target

lb = LabelBinarizer()
y = lb.fit_transform(y)

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

optimizers = ['adam', 'sgd', 'rmsprop']
for optimizer in optimizers:
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, verbose=0)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Optimizer', optimizer)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)