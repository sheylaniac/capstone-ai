import os
import argparse
import datetime
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from src.ai_pipeline.model import build_model
from src.ai_pipeline.components.losses import calculate_multi_objective_loss
from src.ai_pipeline.components.callbacks import ModelCheckpointCallback

def main():
    parser = argparse.ArgumentParser(description="Train multi-output LSTM model with Custom Training Loop")
    parser.add_argument("--version", type=str, default="v1", help="Version of the model to save (e.g. v1, v2)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    workspace = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(workspace, "data", "raw", "datasetdailylogs.csv")
        
    model_save_dir = os.path.join(workspace, "saved_models", args.version)
    artifacts_dir = os.path.join(model_save_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    model_save_path = os.path.join(model_save_dir, "lstmmultioutput.keras")

    print(f"Starting training pipeline for model version: {args.version} ...")
    
    print("Loading raw dataset...")
    df = pd.read_csv(raw_data_path)

    print("Engineering features...")
    df['fatigue_index'] = (df['stress_level'] / 10.0) * 0.4 + (df['study_work_duration'] / (df['sleep_duration'] + 1e-5)) * 0.3 + (1.0 - df['sleep_quality'] / 10.0) * 0.3
    df['cumulative_fatigue'] = df['fatigue_index'].rolling(window=3, min_periods=1).sum()
    df['target_reg'] = df['productivity_score']

    def get_class_label(val):
        if pd.isna(val):
            return np.nan
        if val < 55.0:
            return 0  # At Risk
        elif val <= 67.0:
            return 1  # Steady
        else:
            return 2  # Thriving

    df['target_clf'] = df['target_reg'].apply(get_class_label)
    df = df.dropna(subset=['target_reg', 'target_clf']).reset_index(drop=True)

    print("Splitting dataset chronologically per user...")
    train_dfs, val_dfs, test_dfs = [], [], []
    for user_id, group in df.groupby('user_id'):
        n = len(group)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        train_dfs.append(group.iloc[:n_train])
        val_dfs.append(group.iloc[n_train:n_train+n_val])
        test_dfs.append(group.iloc[n_train+n_val:])

    train_df = pd.concat(train_dfs).reset_index(drop=True)
    val_df = pd.concat(val_dfs).reset_index(drop=True)
    test_df = pd.concat(test_dfs).reset_index(drop=True)

    feature_cols = [
        'sleep_duration', 'sleep_quality', 'study_work_duration', 'break_duration',
        'physical_activity_duration', 'screen_time_duration', 'stress_level', 'mood_score',
        'focus_score', 'task_planned', 'task_completed', 'task_completion_rate',
        'day_of_week', 'month', 'is_weekend', 'cumulative_fatigue'
    ]

    print("Fitting MinMaxScaler and scaling data...")
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(train_df[feature_cols])

    target_scaler = MinMaxScaler()
    target_scaler.fit(train_df[['target_reg']])

    with open(os.path.join(artifacts_dir, "feature_scaler.pkl"), 'wb') as f:
        pickle.dump(feature_scaler, f)
    with open(os.path.join(artifacts_dir, "target_scaler.pkl"), 'wb') as f:
        pickle.dump(target_scaler, f)
    print("Scalers saved successfully.")

    def generate_sequences_for_split(df_split, feat_scaler, targ_scaler, window_size=7):
        X_list, y_reg_list, y_clf_list = [], [], []
        for user_id, group in df_split.groupby('user_id'):
            feats = feat_scaler.transform(group[feature_cols])
            targ_reg = targ_scaler.transform(group[['target_reg']]).flatten()
            targ_clf = tf.keras.utils.to_categorical(group['target_clf'].values, num_classes=3)

            for i in range(len(group) - window_size + 1):
                X_list.append(feats[i : i + window_size])
                y_reg_list.append(targ_reg[i + window_size - 1])
                y_clf_list.append(targ_clf[i + window_size - 1])

        return np.array(X_list), np.array(y_reg_list), np.array(y_clf_list)

    print("Generating 3D sekuens...")
    X_train, y_train_reg, y_train_clf = generate_sequences_for_split(train_df, feature_scaler, target_scaler)
    X_val, y_val_reg, y_val_clf = generate_sequences_for_split(val_df, feature_scaler, target_scaler)

    X_train = X_train.astype(np.float32)
    y_train_reg = y_train_reg.astype(np.float32)
    y_train_clf = y_train_clf.astype(np.float32)
    
    X_val = X_val.astype(np.float32)
    y_val_reg = y_val_reg.astype(np.float32)
    y_val_clf = y_val_clf.astype(np.float32)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, {"out_regression": y_train_reg, "out_classification": y_train_clf}))
    train_ds = train_ds.shuffle(10000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, {"out_regression": y_val_reg, "out_classification": y_val_clf}))
    val_ds = val_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model = build_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    train_mae = tf.keras.metrics.MeanAbsoluteError()
    train_acc = tf.keras.metrics.CategoricalAccuracy()
    val_mae = tf.keras.metrics.MeanAbsoluteError()
    val_acc = tf.keras.metrics.CategoricalAccuracy()

    @tf.function
    def train_step(x, y_reg, y_clf):
        with tf.GradientTape() as tape:
            y_pred_reg, y_pred_clf = model(x, training=True)
            loss_total, loss_reg, loss_clf = calculate_multi_objective_loss(
                y_reg, y_pred_reg, y_clf, y_pred_clf, alpha=1.0, beta=1.0
            )

        gradients = tape.gradient(loss_total, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        y_pred_reg_sq = tf.squeeze(y_pred_reg, axis=-1)
        train_mae.update_state(y_reg, y_pred_reg_sq)
        train_acc.update_state(y_clf, y_pred_clf)
        return loss_total, loss_reg, loss_clf

    @tf.function
    def val_step(x, y_reg, y_clf):
        y_pred_reg, y_pred_clf = model(x, training=False)
        loss_total, loss_reg, loss_clf = calculate_multi_objective_loss(
            y_reg, y_pred_reg, y_clf, y_pred_clf, alpha=1.0, beta=1.0
        )

        y_pred_reg_sq = tf.squeeze(y_pred_reg, axis=-1)
        val_mae.update_state(y_reg, y_pred_reg_sq)
        val_acc.update_state(y_clf, y_pred_clf)
        return loss_total, loss_reg, loss_clf

    log_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(workspace, "logs", "fit", f"{args.version}_{log_time}", "train")
    val_log_dir = os.path.join(workspace, "logs", "fit", f"{args.version}_{log_time}", "val")
    train_writer = tf.summary.create_file_writer(train_log_dir)
    val_writer = tf.summary.create_file_writer(val_log_dir)

    checkpoint_callback = ModelCheckpointCallback(
        filepath=model_save_path, monitor="val_loss", mode="min"
    )

    print("\n--- Starting Custom Training Loop with tf.GradientTape ---")
    for epoch in range(args.epochs):
        train_mae.reset_state()
        train_acc.reset_state()
        val_mae.reset_state()
        val_acc.reset_state()

        total_train_loss = 0.0
        train_batches = 0
        for x_batch, y_batch in train_ds:
            loss_total, _, _ = train_step(x_batch, y_batch['out_regression'], y_batch['out_classification'])
            total_train_loss += loss_total.numpy()
            train_batches += 1
        avg_train_loss = total_train_loss / train_batches

        total_val_loss = 0.0
        val_batches = 0
        for x_batch_val, y_batch_val in val_ds:
            val_loss_total, _, _ = val_step(x_batch_val, y_batch_val['out_regression'], y_batch_val['out_classification'])
            total_val_loss += val_loss_total.numpy()
            val_batches += 1
        avg_val_loss = total_val_loss / val_batches

        train_mae_res = train_mae.result().numpy()
        train_acc_res = train_acc.result().numpy()
        val_mae_res = val_mae.result().numpy()
        val_acc_res = val_acc.result().numpy()

        with train_writer.as_default():
            tf.summary.scalar('loss', avg_train_loss, step=epoch)
            tf.summary.scalar('mae', train_mae_res, step=epoch)
            tf.summary.scalar('accuracy', train_acc_res, step=epoch)
        with val_writer.as_default():
            tf.summary.scalar('loss', avg_val_loss, step=epoch)
            tf.summary.scalar('mae', val_mae_res, step=epoch)
            tf.summary.scalar('accuracy', val_acc_res, step=epoch)

        print(f"Epoch {epoch+1:02d}/{args.epochs} - loss: {avg_train_loss:.4f} - mae: {train_mae_res:.4f} - acc: {train_acc_res:.4f} | val_loss: {avg_val_loss:.4f} - val_mae: {val_mae_res:.4f} - val_acc: {val_acc_res:.4f}")

        checkpoint_callback.check_and_save(avg_val_loss, model)

    print(f"\nTraining pipeline finished. Best model saved at '{model_save_path}'.")

if __name__ == "__main__":
    main()
