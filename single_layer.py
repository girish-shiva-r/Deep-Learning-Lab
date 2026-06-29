"""
Lab 1: Perceptron Learning Implementation (CO1)
Implement the Perceptron learning algorithm from scratch using NumPy.
Focuses on weight update rule and step activation function for binary classification.
"""
import numpy as np
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
# Core Perceptron Components
# ──────────────────────────────────────────────

def step_activation(z):
    """Heaviside Step Activation Function: f(z) = 1 if z >= 0, else 0"""
    return 1 if z >= 0 else 0


def predict(x, weights, bias):
    """Prediction Logic: y_hat = step(w · x + b)"""
    linear_output = x @ weights + bias
    return step_activation(linear_output)


def train_perceptron(X, y, lr=0.1, epochs=10):
    """
    Perceptron Training Loop.

    Weight Update Rule:
        Δw = η * (y - ŷ) * x
        Δb = η * (y - ŷ)

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    lr : float, learning rate (η)
    epochs : int, number of training iterations

    Returns
    -------
    weights : np.ndarray, final learned weights
    bias : float, final learned bias
    """
    weights = np.zeros(X.shape[1])
    bias = 0
    weight_history = []

    for epoch in range(epochs):
        errors = 0
        for i in range(len(X)):
            # 1. Calculate Linear Combination
            linear_output = X[i] @ weights + bias

            # 2. Apply Activation Function
            y_pred = 1 if linear_output >= 0 else 0

            # 3. Compute Update (Error * Learning Rate)
            update = lr * (y[i] - y_pred)

            # 4. Update Weights and Bias
            weights += update * X[i]
            bias += update

            if y[i] != y_pred:
                errors += 1

        weight_history.append((weights.copy(), bias))
        print(f"  Epoch {epoch + 1:>2d}/{epochs} | "
              f"Weights: [{weights[0]:.2f}, {weights[1]:.2f}] | "
              f"Bias: {bias:.2f} | Errors: {errors}")

    return weights, bias, weight_history


def plot_decision_boundary(X, y, weights, bias, title, gate_name):
    """Visualize the decision boundary relative to the data points."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot data points
    for i in range(len(X)):
        color = 'blue' if y[i] == 1 else 'red'
        marker = 'o' if y[i] == 1 else 'x'
        ax.scatter(X[i, 0], X[i, 1], c=color, marker=marker, s=200, zorder=5,
                   edgecolors='black', linewidths=1.5)

    # Decision boundary: w1*x1 + w2*x2 + b = 0  =>  x2 = -(w1*x1 + b) / w2
    x1_range = np.linspace(-0.5, 1.5, 300)
    if abs(weights[1]) > 1e-8:
        x2_boundary = -(weights[0] * x1_range + bias) / weights[1]
        ax.plot(x1_range, x2_boundary, 'g-', linewidth=2,
                label=f'Decision Boundary\n{weights[0]:.2f}·x₁ + {weights[1]:.2f}·x₂ + {bias:.2f} = 0')

        # Shade the positive and negative regions
        ax.fill_between(x1_range, x2_boundary, 2, alpha=0.1, color='blue', label='Class 1 region')
        ax.fill_between(x1_range, x2_boundary, -1, alpha=0.1, color='red', label='Class 0 region')

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f'/home/sasi/Documents/DL/Lab-1/{gate_name}_decision_boundary.png', dpi=150)
    plt.show()
    print(f"  → Plot saved as {gate_name}_decision_boundary.png")


def evaluate_perceptron(X, y, weights, bias):
    """Evaluate perceptron predictions and compute metrics."""
    predictions = [predict(x, weights, bias) for x in X]
    correct = sum(p == t for p, t in zip(predictions, y))
    accuracy = correct / len(y)

    tp = sum(1 for p, t in zip(predictions, y) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(predictions, y) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(predictions, y) if p == 0 and t == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return accuracy, precision, recall, predictions


# ──────────────────────────────────────────────
# Main Execution
# ──────────────────────────────────────────────

def main():
    # ─── Dataset Definitions ───
    datasets = {
        'AND': {
            'X': np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            'y': np.array([0, 0, 0, 1])
        },
        'OR': {
            'X': np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            'y': np.array([0, 1, 1, 1])
        }
    }

    print("=" * 60)
    print("  Lab 1: Perceptron Learning Implementation")
    print("=" * 60)

    results = {}

    for gate_name, data in datasets.items():
        X = data['X']
        y = data['y']

        print(f"\n{'─' * 60}")
        print(f"  Training Perceptron for {gate_name} Gate")
        print(f"{'─' * 60}")
        print(f"  Dataset: {list(zip(X.tolist(), y.tolist()))}")
        print(f"  Learning Rate: 0.1 | Epochs: 10\n")

        weights, bias, history = train_perceptron(X, y, lr=0.1, epochs=10)

        # Output Analysis
        accuracy, precision, recall, predictions = evaluate_perceptron(X, y, weights, bias)
        results[gate_name] = {
            'accuracy': accuracy, 'precision': precision, 'recall': recall
        }

        print(f"\n  ── Output Analysis ({gate_name} Gate) ──")
        print(f"  Dataset Used        : {gate_name}")
        print(f"  Final Weights (w1, w2): ({weights[0]:.4f}, {weights[1]:.4f})")
        print(f"  Final Bias (b)      : {bias:.4f}")
        print(f"  Decision Boundary   : {weights[0]:.4f}·x₁ + {weights[1]:.4f}·x₂ + {bias:.4f} = 0")
        print(f"  Accuracy            : {accuracy:.4f}")
        print(f"  Precision           : {precision:.4f}")
        print(f"  Recall              : {recall:.4f}")

        print(f"\n  Predictions:")
        for i in range(len(X)):
            print(f"    Input: {X[i]} → Predicted: {predictions[i]} | Actual: {y[i]} | "
                  f"{'✓' if predictions[i] == y[i] else '✗'}")

        # Decision Boundary Visualization
        plot_decision_boundary(
            X, y, weights, bias,
            f'Perceptron Decision Boundary ({gate_name} Gate)',
            gate_name
        )

    # ─── XOR Gate (Critical Thinking) ───
    print(f"\n{'─' * 60}")
    print(f"  Critical Thinking: XOR Gate (Non-Linearly Separable)")
    print(f"{'─' * 60}")

    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    print(f"  Dataset: {list(zip(X_xor.tolist(), y_xor.tolist()))}\n")

    weights_xor, bias_xor, _ = train_perceptron(X_xor, y_xor, lr=0.1, epochs=20)
    accuracy_xor, _, _, preds_xor = evaluate_perceptron(X_xor, y_xor, weights_xor, bias_xor)

    print(f"\n  XOR Results:")
    print(f"  Final Weights: ({weights_xor[0]:.4f}, {weights_xor[1]:.4f})")
    print(f"  Final Bias   : {bias_xor:.4f}")
    print(f"  Accuracy     : {accuracy_xor:.4f}")
    for i in range(len(X_xor)):
        print(f"    Input: {X_xor[i]} → Predicted: {preds_xor[i]} | Actual: {y_xor[i]} | "
              f"{'✓' if preds_xor[i] == y_xor[i] else '✗'}")

    print(f"\n  ── Perceptron Convergence Theorem Analysis ──")
    print("  The XOR function is NOT linearly separable — no single hyperplane can")
    print("  separate the classes. According to the Perceptron Convergence Theorem,")
    print("  the algorithm is guaranteed to converge ONLY for linearly separable data.")
    print("  For XOR, the weights oscillate indefinitely without converging, because")
    print("  correcting one misclassification causes another. This demonstrates the")
    print("  fundamental limitation of single-layer perceptrons, motivating the need")
    print("  for multi-layer networks (Lab 2).")

    # ─── Summary Table ───
    print(f"\n{'=' * 60}")
    print(f"  Experiment Summary (Lab 1)")
    print(f"{'=' * 60}")
    print(f"  {'Gate':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12}")
    print(f"  {'─' * 46}")
    for gate, r in results.items():
        print(f"  {gate:<10} {r['accuracy']:<12.4f} {r['precision']:<12.4f} {r['recall']:<12.4f}")


if __name__ == "__main__":
    main()
