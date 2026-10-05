import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

print(f"TensorFlow Version: {tf.__version__}")

max_features = 10000
maxlen = 256

print("Loading data...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print("Pad sequences (samples x time)")
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
epochs_rnn = 5
batch_size_rnn = 32
print("Training RNN model...")
history_rnn = model_rnn.fit(x_train, y_train,
                            batch_size=batch_size_rnn,
                            epochs=epochs_rnn,
                            validation_data=(x_test, y_test))
print("RNN Model Training Finished!")

print("Evaluating RNN model...")
loss_rnn, accuracy_rnn = model_rnn.evaluate(x_test, y_test, verbose=0)

print(f"Test Loss: {loss_rnn:.4f}")
print(f"Test Accuracy: {accuracy_rnn*100:.2f}%")
