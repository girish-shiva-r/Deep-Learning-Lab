import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from sklearn.metrics import classification_report

np.random.seed(42)
tf.random.set_seed(42)

url_train = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TRAIN.tsv"
url_test = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TEST.tsv"

train_data = np.loadtxt(url_train, delimiter="\t")
test_data = np.loadtxt(url_test, delimiter="\t")

y_train = train_data[:, 0]
X_train = train_data[:, 1:]
y_test = test_data[:, 0]
X_test = test_data[:, 1:]

y_train = (y_train + 1) // 2
y_test = (y_test + 1) // 2
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

mean = X_train.mean()
std = X_train.std()
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

model = Sequential([
    Input(shape=(X_train.shape[1], 1)),
    Conv1D(filters=32, kernel_size=5, activation="relu"),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=5, activation="relu"),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.summary()

print("Training the 1D CNN model...")
history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Model Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Model Loss")
plt.legend()
plt.show()

feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.layers[0].output
)

sample_input = X_test[0:1]
feature_maps = feature_extractor.predict(sample_input)

plt.figure(figsize=(10, 6))
plt.suptitle("First 4 Feature Maps from Conv1D Layer")
for i in range(4):
    plt.subplot(4, 1, i + 1)
    plt.plot(feature_maps[0, :, i])
    plt.ylabel(f"Filter {i}")
plt.xlabel("Time Step")
plt.tight_layout()
plt.show()
