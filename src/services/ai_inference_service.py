import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from src.ai_pipeline.components.layers import CustomAttention

original_dense_init = keras.layers.Dense.__init__
def safe_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None) 
    original_dense_init(self, *args, **kwargs) 
keras.layers.Dense.__init__ = safe_dense_init

original_lstm_init = keras.layers.LSTM.__init__
def safe_lstm_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None) 
    original_lstm_init(self, *args, **kwargs) 
keras.layers.LSTM.__init__ = safe_lstm_init


class AIInferenceService:
    def __init__(self, version: str = "v1"):
        self.version = version
        current_dir = os.path.dirname(os.path.abspath(__file__)) 
        src_dir = os.path.dirname(current_dir)                    
        self.workspace = os.path.dirname(src_dir)
        
        self.model_dir = os.path.join(self.workspace, "saved_models", self.version)
        self.model_path = os.path.join(self.model_dir, "lstmmultioutput.keras")
        self.feat_scaler_path = os.path.join(self.model_dir, "artifacts", "feature_scaler.pkl")
        self.targ_scaler_path = os.path.join(self.model_dir, "artifacts", "target_scaler.pkl")
        
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        
        self.feature_cols = [
            'is_weekend', 'sleep_duration', 'study_work_duration', 'break_duration',
            'exercise_duration', 'downtime_duration', 'stress_level', 'mood_score',
            'focus_score', 'task_planned', 'task_completed', 'completion_ratio',
            'fatigue_index', 'cumulative_fatigue'
        ]
        
        self.load_artifacts()

    def load_artifacts(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model version '{self.version}' not found.")
            
        print(f"Loading TensorFlow model version '{self.version}' from {self.model_path}...")
        
        custom_objects = {
            'CustomAttention': CustomAttention
        }

        self.model = tf.keras.models.load_model(
            self.model_path,
            custom_objects=custom_objects,
            compile=False 
        )
        print("Model loaded successfully.")
        
        if not os.path.exists(self.feat_scaler_path) or not os.path.exists(self.targ_scaler_path):
            raise FileNotFoundError("Scaler pkl files not found in version artifacts.")
            
        with open(self.feat_scaler_path, 'rb') as f:
            self.feature_scaler = pickle.load(f)
        with open(self.targ_scaler_path, 'rb') as f:
            self.target_scaler = pickle.load(f)
        print("Scalers loaded successfully.")

    def predict(self, raw_logs: list) -> tuple:

        if self.model is None or self.feature_scaler is None or self.target_scaler is None:
            raise RuntimeError("Model or scalers are not loaded.")

        df_raw = pd.DataFrame(raw_logs)
        
        df_raw['fatigue_index'] = (
            (df_raw['stress_level'] / 8.0)        * 40 +
            (df_raw['downtime_duration'] / 11.21) * 30 +
            (df_raw['study_work_duration'] / 17.405)  * 30
        ).clip(0, 100).round(2)

        df_raw['cumulative_fatigue'] = (
            df_raw['fatigue_index'].ewm(alpha=1-0.9, adjust=False).mean()  # alpha=0.1
        ).clip(0, 100).round(2)

        df_raw['completion_ratio'] = (
            df_raw['task_completed'] / df_raw['task_planned'].replace(0, 1)
        ).clip(0, 1).round(3)
        
        feats_scaled = self.feature_scaler.transform(df_raw[self.feature_cols])
        
        feats_scaled_3d = np.expand_dims(feats_scaled, axis=0).astype(np.float32)
        
        pred_reg, pred_clf = self.model(feats_scaled_3d, training=False)
        pred_reg_np = pred_reg.numpy()
        pred_clf_np = pred_clf.numpy()[0]
        
        reg_score_original = float(self.target_scaler.inverse_transform(pred_reg_np)[0][0])
        
        predicted_class_idx = int(np.argmax(pred_clf_np))
        classes = ['At Risk', 'Steady', 'Thriving']
        predicted_class_name = classes[predicted_class_idx]
        
        probabilities = {
            'At Risk': float(pred_clf_np[0]),
            'Steady': float(pred_clf_np[1]),
            'Thriving': float(pred_clf_np[2])
        }
        
        metrics = {
            'avg_sleep': float(df_raw['sleep_duration'].mean()),
            'avg_sleep_quality': float(df_raw['mood_score'].mean()), 
            'avg_work': float(df_raw['study_work_duration'].mean()),
            'avg_stress': float(df_raw['stress_level'].mean()),
            'avg_screen_time': float(df_raw['downtime_duration'].mean()),
            'avg_downtime': float(df_raw['downtime_duration'].mean()),
            'last_fatigue': float(df_raw['cumulative_fatigue'].iloc[-1])
        }
        
        return reg_score_original, predicted_class_name, probabilities, metrics