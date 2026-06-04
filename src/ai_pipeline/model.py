import tensorflow as tf
from tensorflow.keras import layers
from src.ai_pipeline.components.layers import CustomAttention

def build_model(input_shape=(7, 14), name="multi_output_lstm"):

    inputs = layers.Input(shape=input_shape, name="input_layer")
    
    lstm_out = layers.LSTM(64, return_sequences=True, name="lstm_layer")(inputs)
    
    attention_out = CustomAttention(name="attention_layer")(lstm_out)
    
    dense_shared = layers.Dense(32, activation="relu", name="dense_shared")(attention_out)
    
    # Output 1: Productivity Score Regression (Sigmoid maps output exactly to normalized range [0, 1])
    out_reg = layers.Dense(1, activation="sigmoid", name="out_regression")(dense_shared)
    
    # Output 2: Productivity Status Classification (3-classes Softmax: At Risk, Steady, Thriving)
    out_clf = layers.Dense(3, activation="softmax", name="out_classification")(dense_shared)
    
    model = tf.keras.Model(inputs=inputs, outputs=[out_reg, out_clf], name=name)
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()
