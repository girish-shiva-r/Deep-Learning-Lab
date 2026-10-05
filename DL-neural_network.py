import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from sklearn.metrics import confusion_matrix

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

digit_classifier = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

digit_classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
training_history = digit_classifier.fit(train_images, train_labels, epochs=5, batch_size=32, validation_split=0.1)

test_loss, test_accuracy = digit_classifier.evaluate(test_images, test_labels, verbose=0)
print("\nTest Accuracy:", test_accuracy)

predicted_digits = np.argmax(digit_classifier.predict(test_images), axis=1)
confusion_matrix_result = confusion_matrix(test_labels, predicted_digits)
print("\nConfusion Matrix:")
print(confusion_matrix_result)

misclassified_indices = np.where(test_labels != predicted_digits)[0]
print("First 10 Misclassified Image IDs:")
print(misclassified_indices[:10])

for idx in misclassified_indices[:3]:
    plt.imshow(test_images[idx], cmap='gray')
    plt.title(f"True: {test_labels[idx]}, Predicted: {predicted_digits[idx]}")
    plt.show()