import numpy as np
import matplotlib.pyplot as plt

def train_perceptron(input_data, labels, learning_rate=0.1, num_epochs=20):
    model_weights = np.zeros(input_data.shape[1])
    model_bias = 0
    for epoch in range(num_epochs):
        errors = 0
        for i in range(len(input_data)):
            linear_output = np.dot(input_data[i], model_weights) + model_bias
            predicted_label = 1 if linear_output >= 0 else 0
            update = learning_rate * (labels[i] - predicted_label)
            model_weights += update * input_data[i]
            model_bias += update
            if update != 0:
                errors += 1
        if errors == 0:
            print(f"Converged at epoch {epoch+1}")
            break
    return model_weights, model_bias

input_data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
labels = np.array([0, 0, 0, 1])

model_weights, model_bias = train_perceptron(input_data, labels)
print("Weights:", model_weights)
print("Bias:", model_bias)
print("\nPredictions:")
for sample in input_data:
    prediction = 1 if np.dot(sample, model_weights) + model_bias >= 0 else 0
    print(sample, "->", prediction)

for i in range(len(input_data)):
    if labels[i] == 0:
        plt.scatter(input_data[i,0], input_data[i,1],color='red',marker='o',s=100,label='Class 0' if i == 0 else "")
    else:
        plt.scatter(input_data[i,0], input_data[i,1],color='blue',marker='s',s=100,label='Class 1')

boundary_x = np.linspace(-0.5, 1.5, 100)
if model_weights[1] != 0:
    boundary_y = -(model_weights[0] * boundary_x + model_bias) / model_weights[1]
    plt.plot(boundary_x,boundary_y,color='green',linewidth=2,label='Decision Boundary')

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Perceptron Decision Boundary (AND Gate)")
plt.grid(True)
plt.legend()
plt.show()