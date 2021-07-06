import tensorflow as tf


def f1(y_true, y_pred):
    numerator = tf.reduce_sum(y_true * y_pred, axis=[0, 1, 2])
    denominator = tf.reduce_sum(y_true, axis=[0, 1, 2]) + tf.reduce_sum(y_pred, axis=[0, 1, 2])
    dice = (2. * numerator + 1.) / (denominator + 1.)
    return tf.reduce_mean(dice)
