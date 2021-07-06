import os
from glob import glob
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Dropout, MaxPooling2D, \
    Concatenate, Softmax

tf.config.run_functions_eagerly(True)

drout_rate = 0.3


def double_conv(input, filters):
    conv = Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu', kernel_regularizer='l2')(input)
    conv = BatchNormalization()(conv)
    conv = Dropout(rate=drout_rate)(conv)
    conv = Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu', kernel_regularizer='l2')(conv)
    conv = BatchNormalization()(conv)
    return Dropout(rate=drout_rate)(conv)


def down_layer(input, filters):
    conn = double_conv(input=input, filters=filters)
    output = MaxPooling2D(pool_size=2, strides=2)(conn)
    return output, conn


def up_layer(input, connection, filters):
    trans_conv = Conv2DTranspose(filters=filters, kernel_size=2, strides=2, activation='relu', padding='same',
                                 kernel_regularizer='l2')(input)
    trans_conv = BatchNormalization()(trans_conv)
    trans_conv = Dropout(rate=drout_rate)(trans_conv)
    trans_conv = Concatenate()([connection, trans_conv])
    return double_conv(input=trans_conv, filters=filters // 2)


image_input = Input(shape=(224, 224, 3), name="image_input")
output, conn1 = down_layer(input=image_input, filters=32)
output, conn2 = down_layer(input=output, filters=64)
output, conn3 = down_layer(input=output, filters=128)
output, conn4 = down_layer(input=output, filters=256)
output = double_conv(input=output, filters=512)
output = up_layer(input=output, connection=conn4, filters=256)
output = up_layer(input=output, connection=conn3, filters=128)
output = up_layer(input=output, connection=conn2, filters=64)
output = up_layer(input=output, connection=conn1, filters=32)
output = Conv2D(filters=2, kernel_size=1, padding='same')(output)
probs = Softmax(name="labels")(output)

model = tf.keras.Model([image_input], [probs])
smooth = 1.  # Used to prevent the denominator from 0.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)  # Extend y_true to one dimension.
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def imread(input):
    return tf.image.resize_with_pad(image=tf.image.decode_image(tf.io.read_file(input), channels=3), target_width=224,
                                    target_height=224)


def onehot_labels(input):
    return tf.keras.backend.one_hot(indices=tf.cast(input[:, :, 0] // tf.reduce_max(input), dtype=tf.dtypes.uint8),
                                    num_classes=2)


def data_fetch():
    ph2_derm = sorted(
        glob(pathname="/home/livanosg/projects/mel-seg/data/ph2/data/**/**_Dermoscopic_Image/**.png", recursive=True))
    ph2_lesion = sorted(
        glob(pathname="/home/livanosg/projects/mel-seg/data/ph2/data/**/**_lesion/**.png", recursive=True))
    ph2_zipped = list(zip(ph2_derm, ph2_lesion))
    isic_derm = sorted(glob(f"/home/livanosg/projects/mel-seg/data/ISIC-Archive/Data/Images/**"))[:13784]
    isic_lesion = sorted(glob(pathname=f"/home/livanosg/projects/mel-seg/data/ISIC-Archive/Data/Segmentation/**"))[
                  :13784]
    isic_zipped = list(zip(isic_derm, isic_lesion))
    merged_list = ph2_zipped + isic_zipped
    np.random.shuffle(merged_list)
    merged_list = merged_list[:1000]
    train_data = merged_list[:int(0.7 * len(merged_list))]
    val_data = merged_list[int(0.7 * len(merged_list)):]
    train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
    train_dataset = train_dataset.shuffle(buffer_size=200)
    train_dataset = train_dataset.map(lambda x: ({"image_input": tf.image.per_image_standardization(imread(x[0]))},
                                                 {"labels": onehot_labels(imread(x[1]))}),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(8)
    train_dataset = train_dataset.prefetch(buffer_size=200)

    val_dataset = tf.data.Dataset.from_tensor_slices(val_data)
    val_dataset = val_dataset.shuffle(buffer_size=200)
    val_dataset = val_dataset.map(lambda x: ({"image_input": tf.image.per_image_standardization(imread(x[0]))},
                                             {"labels": onehot_labels(imread(x[1]))}),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(8)
    val_dataset = val_dataset.prefetch(buffer_size=200)

    return train_dataset, val_dataset

if __name__ == '__main__':
    callbacks = [tf.keras.callbacks.TensorBoard(),
                 ]
    train_data, val_data = data_fetch()
    model.compile(optimizer='adam', loss=dice_coef_loss, metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])
    model.fit(train_data, validation_data=val_data, callbacks=callbacks, epochs=5)

    model.save("modell")
