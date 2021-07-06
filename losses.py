import tensorflow as tf
import tensorflow.keras.backend as K

smooth = 1.  # Used to prevent the denominator from 0.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)  # Extend y_true to one dimension.
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def log_dice_loss(y_true, y_pred):
    """both tensors are [b, h, w, classes] and y_pred is in probs form"""
    with tf.name_scope('Weighted_Generalized_Dice_Log_Loss'):
        class_freq = tf.reduce_sum(y_true, axis=[0, 1, 2])
        class_freq = tf.math.maximum(class_freq, 1)
        weights = 1 / (class_freq ** 2)

        numerator = tf.reduce_sum(y_true * y_pred, axis=[0, 1, 2])
        denominator = tf.reduce_sum(y_true + y_pred, axis=[0, 1, 2])
        dice = (2 * weights * (numerator + 1)) / (weights * (denominator + 1))
    return tf.math.reduce_mean(- tf.math.log(dice))
