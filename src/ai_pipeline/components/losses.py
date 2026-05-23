import tensorflow as tf

def calculate_multi_objective_loss(y_reg_true, y_reg_pred, y_clf_true, y_clf_pred, alpha=1.0, beta=1.0):

    y_reg_pred_sq = tf.squeeze(y_reg_pred, axis=-1)
    
    loss_reg = tf.reduce_mean(tf.square(y_reg_true - y_reg_pred_sq))
    
    loss_clf = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_clf_true, y_clf_pred))
    
    loss_total = alpha * loss_reg + beta * loss_clf
    
    return loss_total, loss_reg, loss_clf
