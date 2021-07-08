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


def weighted_crossentropy(y_true, y_pred):  # todo fix weights shape
    class_freq = tf.reduce_sum(y_true, axis=[0, 1, 2], keepdims=True)
    class_freq = tf.math.maximum(class_freq, [1, 1])
    weights = tf.math.pow(tf.math.divide(tf.reduce_sum(class_freq), class_freq), 0.5)
    weights = tf.reduce_sum(y_true * weights, axis=-1)
    return tf.math.reduce_sum(tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred) * weights)


def log_dice_loss(y_true, y_pred):
    """both tensors are [b, h, w, classes] and y_pred is in probs form"""
    with tf.name_scope('Weighted_Generalized_Dice_Log_Loss'):
        class_freq = tf.reduce_sum(y_true, axis=[0, 1, 2])
        class_freq = tf.math.maximum(class_freq, 1)
        weights = 1 / (class_freq ** 2)
        numerator = tf.reduce_sum(y_true * y_pred, axis=[0, 1, 2])
        denominator = tf.reduce_sum(y_true + y_pred, axis=[0, 1, 2])
        dice = (2 * weights * (numerator + 1)) / (weights * (denominator + 1))
    return tf.math.reduce_sum(- tf.math.log(dice))


def custom_loss(y_true, y_pred):
    with tf.name_scope('Custom_loss'):
        dice_loss = log_dice_loss(y_true, y_pred)
        wce_loss = weighted_crossentropy(y_true, y_pred)
        loss = tf.math.multiply(.3, dice_loss) + tf.math.multiply(0.7, wce_loss)
    return loss
