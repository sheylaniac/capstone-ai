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
