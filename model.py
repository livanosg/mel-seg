import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Dropout, MaxPooling2D, \
    Concatenate, Softmax

drout_rate = 0.3


def double_conv(input_tensor, filters):
    conv = Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu', kernel_regularizer='l2')(input_tensor)
    conv = BatchNormalization()(conv)
    conv = Dropout(rate=drout_rate)(conv)
    conv = Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu', kernel_regularizer='l2')(conv)
    conv = BatchNormalization()(conv)
    return Dropout(rate=drout_rate)(conv)


def down_layer(input_tensor, filters):
    conn = double_conv(input_tensor=input_tensor, filters=filters)
    output = MaxPooling2D(pool_size=2, strides=2)(conn)
    return output, conn


def up_layer(input_tensor, connection, filters):
    trans_conv = Conv2DTranspose(filters=filters, kernel_size=2, strides=2, activation='relu', padding='same',
                                 kernel_regularizer='l2')(input_tensor)
    trans_conv = BatchNormalization()(trans_conv)
    trans_conv = Dropout(rate=drout_rate)(trans_conv)
    trans_conv = Concatenate()([connection, trans_conv])
    return double_conv(input_tensor=trans_conv, filters=filters // 2)


def unet():
    image_input = Input(shape=(224, 224, 3), name="image_input")
    with tf.name_scope("Down_1"):
        output, conn1 = down_layer(input_tensor=image_input, filters=32)
    with tf.name_scope("Down_2"):
        output, conn2 = down_layer(input_tensor=output, filters=64)
    with tf.name_scope("Down_3"):
        output, conn3 = down_layer(input_tensor=output, filters=128)
    with tf.name_scope("Down_4"):
        output, conn4 = down_layer(input_tensor=output, filters=256)
    with tf.name_scope("Bridge"):
        output = double_conv(input_tensor=output, filters=512)
    with tf.name_scope("Up_1"):
        output = up_layer(input_tensor=output, connection=conn4, filters=256)
    with tf.name_scope("Up_2"):
        output = up_layer(input_tensor=output, connection=conn3, filters=128)
    with tf.name_scope("Up_3"):
        output = up_layer(input_tensor=output, connection=conn2, filters=64)
    with tf.name_scope("Up_4"):
        output = up_layer(input_tensor=output, connection=conn1, filters=32)
    with tf.name_scope("Output"):
        output = Conv2D(filters=2, kernel_size=1, padding='same')(output)
        probs = Softmax(name="labels")(output)
    return tf.keras.Model([image_input], [probs])


if __name__ == '__main__':
    pass
