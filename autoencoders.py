import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

np.random.seed(42)
time = np.linspace(0, 10, 500)
clean_signal = np.sin(2 * np.pi * 3.2658 * time) + 0.5 * np.sin(2 * np.pi * 3.0 * time)
noise = np.random.normal(0, 0.65368, clean_signal.shape)
noisy_signal = clean_signal + noise

window_size = 50
def create_windows(data, window_size):
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i+window_size])
    return np.array(windows)

clean_windows = create_windows(clean_signal, window_size)
noisy_windows = create_windows(noisy_signal, window_size)

split = int(0.8 * len(clean_windows))
train_noisy = noisy_windows[:split]
train_clean = clean_windows[:split]
test_noisy = noisy_windows[split:]
test_clean = clean_windows[split:]

autoencoder = Sequential([
    Input(shape=(window_size,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(16, activation='relu'),
    Dense(32, activation='relu'),
    Dense(window_size, activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()
print("Training the autoencoder to denoise...")

history = autoencoder.fit(
    train_noisy, train_clean,
    epochs=50,
    batch_size=16,
    validation_data=(test_noisy, test_clean),
    verbose=1
)

denoised_windows = autoencoder.predict(noisy_windows)
denoised_signal = np.zeros_like(clean_signal)
denoised_signal[:len(denoised_windows)] = denoised_windows[:, 0]
denoised_signal[len(denoised_windows):] = denoised_windows[-1, 1:]

plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.title("Original Clean Signal")
plt.plot(time, clean_signal, color='g')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.title("Noisy Signal")
plt.plot(time, noisy_signal, color='r')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.title("Denoised Signal (Autoencoder output)")
plt.plot(time, denoised_signal, color='b')
plt.grid(True)

plt.tight_layout()
plt.show()
