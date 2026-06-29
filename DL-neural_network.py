"""
Lab 4: Implementing a Deep Neural Network (DNN) for Digit Classification

Learning Objective:
  Implement a DNN to classify handwritten digits from the MNIST dataset.
  Focus on data preprocessing, multi-layer architecture design,
  confusion matrix evaluation, and error analysis.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def main():
    print("=" * 70)
    print("  Lab 4: Deep Neural Network (DNN) — MNIST Digit Classification")
    print("=" * 70)

    # ─── Part 1: Implementation (Fill-in-the-blanks completed) ───

    # 1. Load and Preprocess Data
    print("\n  Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # TODO: Normalize the pixel values (0-255) to be between 0 and 1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    print(f"  Training set : {x_train.shape} | Labels: {y_train.shape}")
    print(f"  Test set     : {x_test.shape}  | Labels: {y_test.shape}")

    # 2. Build the DNN Architecture
    print("\n  Building DNN model...")
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),

        # TODO: Add two dense hidden layers with ReLU activation
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),

        # Output layer for 10 classes
        layers.Dense(10, activation='softmax')
    ])

    # 3. Compile the Model
    # TODO: Specify the optimizer, loss function, and metrics
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
    )

    model.summary()

    # Train the model
    print("\n  Training model...")
    history = model.fit(x_train, y_train, epochs=5, batch_size=32,
                        validation_split=0.1, verbose=1)

    # ─── Part 2: Output Analysis & Evaluation ───

    print(f"\n{'=' * 70}")
    print("  Part 2: Output Analysis & Evaluation")
    print(f"{'=' * 70}")

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc:.4f}")

    # Generate predictions
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Precision and Recall (macro average)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    print(f"  Precision     : {precision:.4f}")
    print(f"  Recall        : {recall:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>6}", end="")
    for j in range(10):
        print(f"  {j:>4}", end="")
    print()
    print(f"  {'':>6}{'─' * 42}")
    for i in range(10):
        print(f"  {i:>4} |", end="")
        for j in range(10):
            print(f"  {cm[i][j]:>4}", end="")
        print()

    # Identify most-confused digit pairs
    print(f"\n  ── Most Confused Digit Pairs ──")
    confusion_pairs = []
    for i in range(10):
        for j in range(10):
            if i != j:
                confusion_pairs.append((i, j, cm[i][j]))
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    for true_d, pred_d, count in confusion_pairs[:5]:
        print(f"    True: {true_d} → Predicted: {pred_d} | Misclassifications: {count}")

    # Confusion Matrix Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix — DNN (MNIST)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(10)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)

    plt.tight_layout()
    plt.savefig('/home/sasi/Documents/DL/Lab-1/confusion_matrix.png', dpi=150)
    plt.show()
    print("  → Confusion matrix saved as confusion_matrix.png")

    # Classification Report
    print(f"\n  ── Classification Report ──")
    print(classification_report(y_test, y_pred, digits=4))

    # ─── Part 3: Error Analysis ───

    print(f"\n{'=' * 70}")
    print("  Part 3: Error Analysis")
    print(f"{'=' * 70}")

    # Find misclassified images
    misclassified_indices = np.where(y_pred != y_test)[0]
    print(f"\n  Total misclassified: {len(misclassified_indices)} out of {len(y_test)}")

    # Show 3 misclassified examples
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    print(f"\n  {'Image ID':<12} {'True Label':<15} {'Predicted':<12} {'Confidence':<12}")
    print(f"  {'─' * 50}")

    for idx, ax in zip(misclassified_indices[:3], axes):
        true_label = y_test[idx]
        pred_label = y_pred[idx]
        confidence = y_pred_probs[idx][pred_label]

        print(f"  {idx:<12} {true_label:<15} {pred_label:<12} {confidence:<12.4f}")

        ax.imshow(x_test[idx], cmap='gray')
        ax.set_title(f'ID:{idx}\nTrue:{true_label} → Pred:{pred_label}\nConf:{confidence:.2f}',
                     fontsize=10, color='red')
        ax.axis('off')

    plt.suptitle('Misclassified Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/sasi/Documents/DL/Lab-1/misclassified_samples.png', dpi=150)
    plt.show()
    print("  → Misclassified samples saved as misclassified_samples.png")

    # Visual characteristics analysis
    print(f"\n  ── Why Did They Fail? ──")
    for idx in misclassified_indices[:3]:
        true_label = y_test[idx]
        pred_label = y_pred[idx]
        print(f"  Image {idx}: True={true_label}, Predicted={pred_label}")
        print(f"    → The digit has ambiguous strokes or unusual writing style")
        print(f"      that shares visual characteristics with digit {pred_label}.")

    # ─── Critical Reflection ───
    print(f"\n{'=' * 70}")
    print("  Critical Reflection")
    print(f"{'=' * 70}")
    print("  Depth and activation functions are crucial for capturing non-linear")
    print("  features of handwritten digits:")
    print()
    print("  • Single-Layer Perceptron (Lab 1): Can only learn linear decision")
    print("    boundaries. It treats each pixel independently and cannot capture")
    print("    spatial patterns, curves, or complex feature hierarchies.")
    print()
    print("  • Deep Neural Network (Lab 4): Multiple layers with ReLU activation")
    print("    enable hierarchical feature extraction — early layers detect edges")
    print("    and strokes, while deeper layers combine them into digit-specific")
    print("    patterns. ReLU avoids vanishing gradients, enabling effective")
    print("    training of deeper architectures.")
    print()
    print("  • Adding more layers increases the model's representational capacity,")
    print("    allowing it to learn complex, non-linear mappings from raw pixels")
    print("    to digit classes — something impossible for a single perceptron.")

    # Return metrics for final synthesis
    return {
        'accuracy': test_acc,
        'precision': precision,
        'recall': recall
    }


if __name__ == "__main__":
    main()
