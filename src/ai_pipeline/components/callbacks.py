import os
import tensorflow as tf


class ModelCheckpointCallback:

    def __init__(self, filepath, monitor="val_loss", mode="min"):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best = float('inf') if mode == "min" else -float('inf')

    def check_and_save(self, current_value, model):

        improved = False
        if self.mode == "min":
            if current_value < self.best:
                self.best = current_value
                improved = True
        else:
            if current_value > self.best:
                self.best = current_value
                improved = True

        if improved:
            dir_name = os.path.dirname(self.filepath)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            model.save(self.filepath)
            print(f"   [Callback] Model terbaik disimpan ke '{self.filepath}' dengan {self.monitor}: {current_value:.4f}")
            return True
        return False


class EarlyStoppingLRCallback(tf.keras.callbacks.Callback):
    """Custom callback combining early stopping and learning rate reduction.

    Monitors val_loss each epoch; saves the best model, reduces LR when
    stalled, and signals `stop_training` when patience is exhausted.
    """

    def __init__(self, model, optimizer, patience=5, lr_patience=5,
                 lr_factor=0.5, lr_min=1e-6, save_path='best_model.keras'):
        super().__init__()
        self.tracked_model = model
        self.optimizer = optimizer
        self.patience = patience
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_min = lr_min
        self.save_path = save_path
        self.best_val_loss = float('inf')
        self.wait = 0
        self.lr_wait = 0
        self.stop_training = False

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        if val_loss is None:
            return

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.wait = 0
            self.lr_wait = 0
            self.tracked_model.save(self.save_path)
            print(f"   [Callback] Model terbaik disimpan. val_loss: {val_loss:.4f}")
        else:
            self.wait += 1
            self.lr_wait += 1
            print(f"   [Callback] Tidak ada improvement. Patience: {self.wait}/{self.patience}")

            if self.lr_wait >= self.lr_patience:
                current_lr = float(self.optimizer.learning_rate)
                new_lr = max(current_lr * self.lr_factor, self.lr_min)
                if new_lr < current_lr:
                    self.optimizer.learning_rate.assign(new_lr)
                    print(f"   [Callback] LR turun -> {new_lr:.6f}")
                self.lr_wait = 0

            if self.wait >= self.patience:
                self.stop_training = True
                print(f"   [Callback] Early stopping di epoch {epoch+1}. "
                      f"Val_loss tidak membaik selama {self.patience} epoch.")
