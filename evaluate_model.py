import os
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.ai_pipeline.components.layers import CustomAttention

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained LSTM model on test set")
    parser.add_argument("--version", type=str, default="v1", help="Version of the model to evaluate")
    args = parser.parse_args()

    workspace = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(workspace, "saved_models", args.version, "lstmmultioutput.keras")
    
    x_test_path = os.path.join(workspace, "data", "processed", "x_test.npy")
    y_test_reg_path = os.path.join(workspace, "data", "processed", "y_test_reg.npy")
    y_test_clf_path = os.path.join(workspace, "data", "processed", "y_test_clf.npy")

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        print("Please train the model first using: python -m src.ai_pipeline.train")
        return

    if not os.path.exists(x_test_path):
        print(f"ERROR: Test split files not found at {x_test_path}")
        print("Please run training first to generate test files.")
        return

    print("=" * 60)
    print(f"EVALUATING MODEL VERSION: {args.version}")
    print("=" * 60)

    print("Loading test dataset splits...")
    X_test = np.load(x_test_path)
    y_test_reg = np.load(y_test_reg_path)
    y_test_clf = np.load(y_test_clf_path)

    print("Loading TensorFlow model...")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'CustomAttention': CustomAttention}
    )

    print("Running predictions on test set...")
    pred_reg, pred_clf = model.predict(X_test, verbose=0)
    pred_reg_sq = pred_reg.squeeze()

    # Calculate Regression MAE
    mae_metric = tf.keras.metrics.MeanAbsoluteError()
    mae_metric.update_state(y_test_reg, pred_reg_sq)
    mae_val = mae_metric.result().numpy()

    # Calculate Classification Accuracy
    acc_metric = tf.keras.metrics.CategoricalAccuracy()
    acc_metric.update_state(y_test_clf, pred_clf)
    acc_val = acc_metric.result().numpy()

    print("\n" + "=" * 60)
    print("                    EVALUATION RESULTS")
    print("=" * 60)
    print(f"Classification Accuracy : {acc_val * 100:.2f}% (Target: >= 85.00%)")
    print(f"Regression MAE          : {mae_val:.4f} (Target: <= 0.0200)")
    print("-" * 60)

    # Check targets
    if acc_val >= 0.85 and mae_val <= 0.02:
        print("STATUS: PERFORMANCE TARGET SUCCESSFULLY MET! [SUCCESS]")
    else:
        print("STATUS: PERFORMANCE TARGET FAILED. TRY ADJUSTING HYPERPARAMETERS.")
    print("=" * 60)

    # Confusion Matrix
    print("\nGenerating Confusion Matrix...")
    y_true_classes = np.argmax(y_test_clf, axis=1)
    y_pred_classes = np.argmax(pred_clf, axis=1)

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['At Risk', 'Steady', 'Thriving'])

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix - Model {args.version}')
    plt.tight_layout()

    # Simpan gambar
    plot_path = os.path.join(workspace, "saved_models", args.version, "confusion_matrix.png")
    plt.savefig(plot_path)
    print(f"Confusion matrix saved to '{plot_path}'")

    history_path = os.path.join(workspace, "saved_models", args.version, "history_metrics.json")
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history_metrics = json.load(f)

        epochs_range = range(1, len(history_metrics['train_loss']) + 1)
        fig, ax = plt.subplots(1, 2, figsize=(16, 5))

        # Kurva loss
        ax[0].plot(epochs_range, history_metrics['train_loss'], label='Train Loss')
        ax[0].plot(epochs_range, history_metrics['val_loss'], label='Val Loss')
        ax[0].set_title('Kurva Loss Pelatihan')
        ax[0].set_xlabel('Epoch')
        ax[0].legend()

        # Kurva akurasi
        ax[1].plot(epochs_range, history_metrics['train_acc'], label='Train Accuracy')
        ax[1].plot(epochs_range, history_metrics['val_acc'], label='Val Accuracy')
        ax[1].set_title('Kurva Akurasi Pelatihan')
        ax[1].set_xlabel('Epoch')
        ax[1].legend()

        plt.tight_layout()
        plot_path = os.path.join(workspace, "saved_models", args.version, "training_curves.png")
        plt.savefig(plot_path)
        print(f"Training curves saved to '{plot_path}'")
    else:
        print("History metrics not found, skipping training curves.")

if __name__ == "__main__":
    main()
