import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

xor_input = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
xor_output = np.array([0,1,1,0])

neural_network = Sequential([
    Input(shape=(2,)),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

neural_network.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

training_history = neural_network.fit(
    xor_input,
    xor_output,
    epochs=20,
    verbose=1
)

loss, accuracy = neural_network.evaluate(xor_input, xor_output, verbose=0)
print("\nFinal Loss:", loss)
print("Final Accuracy:", accuracy)

network_predictions = (neural_network.predict(xor_input) > 0.5).astype(int)
print("\nPredictions:")
for i in range(len(xor_input)):
    print(xor_input[i], "->", network_predictions[i][0])

config_list = [
    ("sigmoid", 0.01),
    ("sigmoid", 0.1),
    ("relu", 0.01),
    ("relu", 0.1)
]

for activation_func, learning_rate in config_list:
    comparison_model = Sequential([
        Input(shape=(2,)),
        Dense(8, activation=activation_func),
        Dense(1, activation='sigmoid')
    ])
    comparison_model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    comparison_history = comparison_model.fit(
        xor_input,
        xor_output,
        epochs=20,
        verbose=0
    )
    eval_loss, eval_acc = comparison_model.evaluate(xor_input, xor_output, verbose=0)
    print(
        f"Activation={activation_func}, "
        f"LR={learning_rate}, "
        f"Loss={eval_loss:.4f}, "
        f"Accuracy={eval_acc:.4f}"
    )
