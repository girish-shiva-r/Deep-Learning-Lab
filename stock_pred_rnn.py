import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print(f"TensorFlow Version: {tf.__version__}")
ticker = "AAPL"
print(f"Downloading historical data for {ticker}...")
df = yf.download(ticker, start="2019-01-01", end="2024-01-01", auto_adjust=False)
data = df[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_sequences(dataset, lookback=10):
    X, y = [], []
    for i in range(lookback, len(dataset)):
        X.append(dataset[i - lookback : i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

LOOKBACK = 10
X, y = create_sequences(scaled_data, lookback=LOOKBACK)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model_ts = Sequential([
    SimpleRNN(32, return_sequences=False, input_shape=(LOOKBACK, 1)),
    Dense(16, activation='relu'),
    Dense(1)
])
model_ts.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model_ts.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
epochs = 30
batch_size = 16

print("\nTraining SimpleRNN on real-world stock data...")
history = model_ts.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test),
    verbose=1
)

predictions_scaled = model_ts.predict(X_test)
predictions = scaler.inverse_transform(predictions_scaled)
actuals = scaler.inverse_transform(y_test.reshape(-1, 1))
mse = mean_squared_error(actuals, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actuals, predictions)
mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
r2 = r2_score(actuals, predictions)

print("\n" + "="*40)
print("          EVALUATION METRICS (TEST SET)          ")
print("="*40)
print(f"Root Mean Squared Error (RMSE) : ${rmse:.2f}")
print(f"Mean Absolute Error (MAE)      : ${mae:.2f}")
print(f"Mean Absolute Percentage Error : {mape:.2f}%")
print(f"R^2 Score (Variance Explained) : {r2:.4f}")
print("="*40)

print("\nFirst 5 Sample Predictions (Predicted vs Actual AAPL Close Price):")
for i in range(5):
    print(f"Day {i+1}: Predicted = ${predictions[i][0]:.2f} | Actual = ${actuals[i][0]:.2f}")
