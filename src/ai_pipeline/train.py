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
from src.ai_pipeline.components.callbacks import ModelCheckpointCallback, EarlyStoppingLRCallback


def main():
    parser = argparse.ArgumentParser(description="Train multi-output LSTM model with Custom Training Loop")
    parser.add_argument("--version", type=str, default="v1", help="Version of the model to save (e.g. v1, v2)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for regression loss (alpha)")
    parser.add_argument("--beta", type=float, default=3.0, help="Weight for classification loss (beta)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()

    workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_data_path = os.path.join(workspace, "data", "raw", "final_dataset_model_ready.csv")
        
    model_save_dir = os.path.join(workspace, "saved_models", args.version)
    artifacts_dir = os.path.join(model_save_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    model_save_path = os.path.join(model_save_dir, "lstmmultioutput.keras")

    print(f"Starting training pipeline for model version: {args.version} (alpha={args.alpha}, beta={args.beta}, patience={args.patience}) ...")
    
    print("Loading raw dataset...")
    df = pd.read_csv(raw_data_path)

    print("Engineering features...")
    df['target_reg'] = df['productivity_score']
    df['target_clf'] = df['productivity_label']
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
        'is_weekend', 'sleep_duration', 'study_work_duration', 'break_duration',
        'exercise_duration', 'downtime_duration', 'stress_level', 'mood_score',
        'focus_score', 'task_planned', 'task_completed', 'completion_ratio',
        'fatigue_index', 'cumulative_fatigue'
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
    X_test, y_test_reg, y_test_clf = generate_sequences_for_split(test_df, feature_scaler, target_scaler)

    X_train = X_train.astype(np.float32)
    y_train_reg = y_train_reg.astype(np.float32)
    y_train_clf = y_train_clf.astype(np.float32)
    
    X_val = X_val.astype(np.float32)
    y_val_reg = y_val_reg.astype(np.float32)
    y_val_clf = y_val_clf.astype(np.float32)

    X_test = X_test.astype(np.float32)
    y_test_reg = y_test_reg.astype(np.float32)
    y_test_clf = y_test_clf.astype(np.float32)

    processed_dir = os.path.join(workspace, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    np.save(os.path.join(processed_dir, "x_test.npy"), X_test)
    np.save(os.path.join(processed_dir, "y_test_reg.npy"), y_test_reg)
    np.save(os.path.join(processed_dir, "y_test_clf.npy"), y_test_clf)
    test_df.to_csv(os.path.join(processed_dir, "test_raw_dataframe.csv"), index=False)
    print("Test split datasets exported successfully to data/processed/.")

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, {"out_regression": y_train_reg, "out_classification": y_train_clf}))
    train_ds = train_ds.shuffle(10000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, {"out_regression": y_val_reg, "out_classification": y_val_clf}))
    val_ds = val_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model = build_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    alpha_loss = tf.constant(args.alpha, dtype=tf.float32)
    beta_loss = tf.constant(args.beta, dtype=tf.float32)

    train_mae = tf.keras.metrics.MeanAbsoluteError()
    train_acc = tf.keras.metrics.CategoricalAccuracy()
    val_mae = tf.keras.metrics.MeanAbsoluteError()
    val_acc = tf.keras.metrics.CategoricalAccuracy()

    @tf.function
    def train_step(x, y_reg, y_clf):
        with tf.GradientTape() as tape:
            y_pred_reg, y_pred_clf = model(x, training=True)
            loss_total, loss_reg, loss_clf = calculate_multi_objective_loss(
                y_reg, y_pred_reg, y_clf, y_pred_clf, alpha=alpha_loss, beta=beta_loss
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
            y_reg, y_pred_reg, y_clf, y_pred_clf, alpha=alpha_loss, beta=beta_loss
        )

        y_pred_reg_sq = tf.squeeze(y_pred_reg, axis=-1)
        val_mae.update_state(y_reg, y_pred_reg_sq)
        val_acc.update_state(y_clf, y_pred_clf)
        return loss_total, loss_reg, loss_clf

    log_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(workspace, "logs", "gradient_tape", log_time, "train")
    val_log_dir = os.path.join(workspace, "logs", "gradient_tape", log_time, "val")
    train_writer = tf.summary.create_file_writer(train_log_dir)
    val_writer = tf.summary.create_file_writer(val_log_dir)

    callback = EarlyStoppingLRCallback(model=model, optimizer=optimizer, patience=args.patience, lr_patience=args.patience, save_path=model_save_path)
    history_metrics = {'train_loss': [], 'train_mae': [], 'train_acc': [], 'val_loss': [], 'val_mae': [], 'val_acc': []}

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

        # Save history metrics
        history_metrics['train_loss'].append(float(avg_train_loss))
        history_metrics['train_mae'].append(float(train_mae_res))
        history_metrics['train_acc'].append(float(train_acc_res))
        history_metrics['val_loss'].append(float(avg_val_loss))
        history_metrics['val_mae'].append(float(val_mae_res))
        history_metrics['val_acc'].append(float(val_acc_res))

        with train_writer.as_default():
            tf.summary.scalar('loss', avg_train_loss, step=epoch)
            tf.summary.scalar('mae', train_mae_res, step=epoch)
            tf.summary.scalar('accuracy', train_acc_res, step=epoch)
        with val_writer.as_default():
            tf.summary.scalar('loss', avg_val_loss, step=epoch)
            tf.summary.scalar('mae', val_mae_res, step=epoch)
            tf.summary.scalar('accuracy', val_acc_res, step=epoch)

        print(f"Epoch {epoch+1:02d}/{args.epochs} - loss: {avg_train_loss:.4f} - mae: {train_mae_res:.4f} - acc: {train_acc_res:.4f} | val_loss: {avg_val_loss:.4f} - val_mae: {val_mae_res:.4f} - val_acc: {val_acc_res:.4f}")

        callback.on_epoch_end(epoch, logs={'val_loss': avg_val_loss})

        if callback.stop_training:
            break

    # Save history to json file
    import json
    history_json_path = os.path.join(model_save_dir, "history_metrics.json")
    with open(history_json_path, 'w') as f:
        json.dump(history_metrics, f)
    print(f"Training history saved to '{history_json_path}'")

    print(f"\nTraining pipeline finished. Best model saved at '{model_save_path}' (Best Val Loss: {callback.best_val_loss:.4f}).")

if __name__ == "__main__":
    main()
