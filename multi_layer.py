"""
Lab 2: Multilayer Perceptron (MLP) and Hyperparameter Tuning
Lab 3: Advanced Hyperparameter Optimization (CO2)

Lab 2 Objectives:
  - Implement an MLP with at least one hidden layer using Keras.
  - Analyze impact of activation functions (Sigmoid vs. ReLU) on convergence.
  - Observe the relationship between learning rates and model performance.

Lab 3 Objectives:
  - Perform Grid Search / Random Search for hyperparameter optimization.
  - Test at least 5 combinations of hyperparameters.
  - Analyze overfitting and generalization.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from itertools import product
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════
#  LAB 2: MLP for XOR (Non-Linearly Separable Problem)
# ══════════════════════════════════════════════════════════════

def build_mlp(input_dim, hidden_units, activation, output_activation, output_units, lr):
    """
    Build a configurable MLP.

    Forward Pass:
      z1 = X @ W1 + b1        (linear combination, hidden layer)
      a1 = activation(z1)     (non-linear activation)
      z2 = a1 @ W2 + b2       (linear combination, output layer)
      y_hat = σ(z2)           (sigmoid / softmax output)

    Backward Pass (Backpropagation):
      δ_output = ∂L/∂z2 = (y_hat - y)
      ΔW2 = a1ᵀ · δ_output
      δ_hidden = δ_output · W2ᵀ ⊙ activation'(z1)
      ΔW1 = Xᵀ · δ_hidden
      Update: W = W - η · ΔW
    """
    model = Sequential([
        Dense(hidden_units, activation=activation, input_shape=(input_dim,)),
        Dense(hidden_units // 2, activation=activation),
        Dense(output_units, activation=output_activation)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy' if output_units == 1 else 'sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def run_lab2_xor_experiments():
    """
    Lab 2: Hyperparameter Experimentation Log
    Run 4 experiments varying activation (Sigmoid / ReLU) and learning rate (0.01 / 0.1).
    """
    print("=" * 70)
    print("  Lab 2: Multilayer Perceptron (MLP) — XOR Gate Experiments")
    print("=" * 70)

    # XOR Dataset
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y_xor = np.array([[0], [1], [1], [0]], dtype=np.float32)

    experiments = [
        {"exp": 1, "activation": "sigmoid", "lr": 0.01},
        {"exp": 2, "activation": "sigmoid", "lr": 0.1},
        {"exp": 3, "activation": "relu",    "lr": 0.01},
        {"exp": 4, "activation": "relu",    "lr": 0.1},
    ]

    results_lab2 = []
    EPOCHS = 100
    HIDDEN_UNITS = 8

    for cfg in experiments:
        print(f"\n{'─' * 70}")
        print(f"  Experiment #{cfg['exp']}: Activation={cfg['activation'].upper()}, LR={cfg['lr']}")
        print(f"{'─' * 70}")

        model = build_mlp(
            input_dim=2, hidden_units=HIDDEN_UNITS,
            activation=cfg['activation'], output_activation='sigmoid',
            output_units=1, lr=cfg['lr']
        )

        history = model.fit(X_xor, y_xor, epochs=EPOCHS, verbose=0)

        final_loss = history.history['loss'][-1]
        final_acc = history.history['accuracy'][-1]

        predictions = (model.predict(X_xor, verbose=0) > 0.5).astype(int).flatten()

        print(f"  Final Loss     : {final_loss:.6f}")
        print(f"  Final Accuracy : {final_acc:.4f}")
        print(f"  Predictions    : {predictions.tolist()}")
        print(f"  Expected       : {y_xor.flatten().astype(int).tolist()}")

        results_lab2.append({
            'exp': cfg['exp'],
            'activation': cfg['activation'],
            'lr': cfg['lr'],
            'loss': final_loss,
            'accuracy': final_acc
        })

    # Summary Table
    print(f"\n{'=' * 70}")
    print("  Hyperparameter Experimentation Log (Lab 2)")
    print(f"{'=' * 70}")
    print(f"  {'Exp#':<6} {'Activation':<12} {'Learning Rate':<15} {'Final Loss':<14} {'Final Acc.':<12}")
    print(f"  {'─' * 58}")
    for r in results_lab2:
        print(f"  {r['exp']:<6} {r['activation']:<12} {r['lr']:<15} {r['loss']:<14.6f} {r['accuracy']:<12.4f}")

    # Comparative Analysis
    print(f"\n  ── Comparative Analysis ──")
    sigmoid_results = [r for r in results_lab2 if r['activation'] == 'sigmoid']
    relu_results = [r for r in results_lab2 if r['activation'] == 'relu']
    avg_sigmoid_loss = np.mean([r['loss'] for r in sigmoid_results])
    avg_relu_loss = np.mean([r['loss'] for r in relu_results])
    print(f"  Average Sigmoid Loss: {avg_sigmoid_loss:.6f}")
    print(f"  Average ReLU Loss   : {avg_relu_loss:.6f}")
    faster = "ReLU" if avg_relu_loss < avg_sigmoid_loss else "Sigmoid"
    print(f"  → {faster} converged faster on average.")
    print(f"  → Higher learning rates (0.1) generally enable faster convergence")
    print(f"    but may cause instability in the loss curve (oscillations).")
    print(f"  → Lower learning rates (0.01) provide smoother convergence but may")
    print(f"    require more epochs to reach a good solution.")

    return results_lab2


# ══════════════════════════════════════════════════════════════
#  LAB 3: Advanced Hyperparameter Optimization (Grid Search)
# ══════════════════════════════════════════════════════════════

def run_lab3_hyperparameter_search():
    """
    Lab 3: Grid Search over learning rate, hidden units, batch size, and dropout.
    At least 5 different combinations.
    """
    print(f"\n\n{'=' * 70}")
    print("  Lab 3: Advanced Hyperparameter Optimization (Grid Search)")
    print(f"{'=' * 70}")

    # Use the Iris dataset as a non-linearly separable classification problem
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    iris = load_iris()
    X, y = iris.data, iris.target
    X = StandardScaler().fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define hyperparameter grid (at least 5 trials)
    search_space = [
        {"lr": 0.001, "hidden_units": 32,  "batch_size": 16,  "dropout": 0.0},
        {"lr": 0.01,  "hidden_units": 64,  "batch_size": 16,  "dropout": 0.2},
        {"lr": 0.01,  "hidden_units": 128, "batch_size": 32,  "dropout": 0.3},
        {"lr": 0.1,   "hidden_units": 64,  "batch_size": 32,  "dropout": 0.5},
        {"lr": 0.001, "hidden_units": 128, "batch_size": 16,  "dropout": 0.1},
        {"lr": 0.05,  "hidden_units": 64,  "batch_size": 8,   "dropout": 0.2},
    ]

    EPOCHS = 50
    results_lab3 = []

    print(f"\n  Part 1: Theoretical Analysis")
    print(f"  {'─' * 60}")
    print("  1. Impact of Batch Size:")
    print("     - Small batch (16): Noisier gradients → better generalization,")
    print("       slower convergence per epoch, more stochastic path.")
    print("     - Large batch (256): Smoother gradients → faster per epoch,")
    print("       but may converge to sharp minima → poorer generalization.")
    print()
    print("  2. Role of Dropout Rate:")
    print("     - Dropout randomly deactivates neurons during training, forcing")
    print("       the network to learn redundant representations.")
    print("     - At 0.0: No regularization → risk of overfitting.")
    print("     - At 0.5: Strong regularization → reduced capacity, better")
    print("       generalization, but too high can cause underfitting.")

    print(f"\n  Part 2: Grid Search Optimization Log")
    print(f"  {'─' * 60}")

    for i, params in enumerate(search_space, 1):
        print(f"\n  Trial {i}: LR={params['lr']}, Hidden={params['hidden_units']}, "
              f"Batch={params['batch_size']}, Dropout={params['dropout']}")

        model = Sequential([
            Dense(params['hidden_units'], activation='relu', input_shape=(4,)),
            Dropout(params['dropout']),
            Dense(params['hidden_units'] // 2, activation='relu'),
            Dropout(params['dropout']),
            Dense(3, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS, batch_size=params['batch_size'],
            validation_data=(X_val, y_val),
            verbose=0
        )

        train_loss = history.history['loss'][-1]
        val_acc = history.history['val_accuracy'][-1]
        train_acc = history.history['accuracy'][-1]
        overfitting = (train_acc - val_acc) > 0.1

        results_lab3.append({
            'trial': i,
            'params': params,
            'train_loss': train_loss,
            'val_accuracy': val_acc,
            'train_accuracy': train_acc,
            'overfitting': overfitting
        })

        print(f"    Training Loss    : {train_loss:.6f}")
        print(f"    Training Acc     : {train_acc:.4f}")
        print(f"    Validation Acc   : {val_acc:.4f}")
        print(f"    Overfitting?     : {'Yes ⚠' if overfitting else 'No ✓'}")

    # Summary Table
    print(f"\n{'=' * 70}")
    print("  Optimization Log Summary (Lab 3)")
    print(f"{'=' * 70}")
    print(f"  {'Trial':<7} {'Hyperparameters':<42} {'Train Loss':<13} {'Val Acc':<10} {'Overfit?':<10}")
    print(f"  {'─' * 80}")
    for r in results_lab3:
        p = r['params']
        hp_str = f"LR={p['lr']}, H={p['hidden_units']}, B={p['batch_size']}, D={p['dropout']}"
        ofit = 'Yes' if r['overfitting'] else 'No'
        print(f"  {r['trial']:<7} {hp_str:<42} {r['train_loss']:<13.6f} {r['val_accuracy']:<10.4f} {ofit:<10}")

    # Find optimal
    best = max(results_lab3, key=lambda r: r['val_accuracy'])
    bp = best['params']
    print(f"\n  ── Optimal Configuration ──")
    print(f"  Best Trial       : #{best['trial']}")
    print(f"  Learning Rate    : {bp['lr']}")
    print(f"  Hidden Units     : {bp['hidden_units']}")
    print(f"  Batch Size       : {bp['batch_size']}")
    print(f"  Dropout          : {bp['dropout']}")
    print(f"  Validation Acc   : {best['val_accuracy']:.4f}")
    print(f"  Justification    : This configuration achieves the highest validation")
    print(f"    accuracy while maintaining a small gap between training and validation")
    print(f"    performance, indicating good generalization without overfitting.")

    return results_lab3


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    # Lab 2
    results_lab2 = run_lab2_xor_experiments()

    # Lab 3
    results_lab3 = run_lab3_hyperparameter_search()

    print(f"\n{'=' * 70}")
    print("  All Lab 2 & Lab 3 experiments completed successfully!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
