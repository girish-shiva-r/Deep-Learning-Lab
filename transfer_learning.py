import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

train_size = 5000
test_size = 1000
x_train, y_train = x_train[:train_size], y_train[:train_size]
x_test, y_test = x_test[:test_size], y_test[:test_size]

def preprocess_image(image, label):
    image = tf.image.resize(image, (96, 96))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.map(preprocess_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.map(preprocess_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)

print("Loading Pretrained MobileNetV2 model...")
base_model = MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
print("\nTraining the new layers (Feature Extraction Phase)...")

history = model.fit(
    train_dataset,
    epochs=5,
    validation_data=test_dataset
)

loss, accuracy = model.evaluate(test_dataset, verbose=0)
print(f"\nTest Accuracy after Transfer Learning: {accuracy*100:.2f}%")

print("\nUnfreezing base model layers for Fine-tuning...")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history_finetune = model.fit(
    train_dataset,
    epochs=3,
    validation_data=test_dataset
)

loss, accuracy = model.evaluate(test_dataset, verbose=0)
print(f"\nTest Accuracy after Fine-tuning: {accuracy*100:.2f}%")

plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'] + history_finetune.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'] + history_finetune.history['val_accuracy'], label='Validation Accuracy')
plt.axvline(x=4.5, color='r', linestyle='--', label='Start Fine-tuning')
plt.title('Transfer Learning Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
