from glob import glob
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


def imread(file_path, channels):
    image = tf.image.resize_with_pad(image=tf.image.decode_image(tf.io.read_file(file_path), channels=channels),
                                     target_width=224,
                                     target_height=224)
    return tf.reshape(image, [224, 224, channels])


def onehot_labels(input_tensor):
    return tf.keras.backend.one_hot(indices=tf.cast(tf.squeeze(input_tensor, axis=-1) // tf.reduce_max(input_tensor), dtype=tf.dtypes.uint8),
                                    num_classes=2)


def augm(image, label):
    # random brightness adjustment illumination
    image["image_input"] = tf.image.random_brightness(image["image_input"], 0.3)
    # random contrast adjustment
    image["image_input"] = tf.image.random_contrast(image["image_input"], 0.2, 0.5)

    if tf.random.uniform(()) > 0.5:
        image["image_input"] = tf.image.flip_left_right(image["image_input"])
        label["labels"] = tf.image.flip_left_right(label["labels"])
    if tf.random.uniform(()) > 0.5:
        image["image_input"] = tf.image.flip_up_down(image["image_input"])
        label["labels"] = tf.image.flip_up_down(label["labels"])

    rot_factor = tf.cast(tf.random.uniform(shape=[], maxval=12, dtype=tf.int32), tf.float32)
    angle = np.pi / 12 * rot_factor
    image["image_input"] = tfa.image.rotate(image["image_input"], angle)
    label["labels"] = tfa.image.rotate(label["labels"], angle)

    return image, label


def get_datasets():
    ph2_derm = sorted(
        glob(pathname="/home/livanosg/projects/mel-seg/data/ph2/data/**/**_Dermoscopic_Image/**.png", recursive=True))
    ph2_mask = sorted(
        glob(pathname="/home/livanosg/projects/mel-seg/data/ph2/data/**/**_lesion/**.png", recursive=True))
    ph2_zipped = list(zip(ph2_derm, ph2_mask))
    dermofit_derm = sorted(
        glob(pathname="/home/livanosg/projects/mel-seg/data/dermofit/data/**/**.png", recursive=True))
    dermofit_mask = sorted(
        glob(pathname="/home/livanosg/projects/mel-seg/data/dermofit/data/**/**mask.png", recursive=True))
    dermofit_derm = list(set(dermofit_derm) - set(dermofit_mask))
    dermofit_zipped = list(zip(dermofit_derm, dermofit_mask))
    isic_derm = sorted(glob(f"/home/livanosg/projects/mel-seg/data/ISIC-Archive/Data/Images/**"))[:13784]
    isic_mask = sorted(glob(pathname=f"/home/livanosg/projects/mel-seg/data/ISIC-Archive/Data/Segmentation/**"))[:13784]
    isic_zipped = list(zip(isic_derm, isic_mask))
    merged_list = ph2_zipped + isic_zipped + dermofit_zipped
    np.random.shuffle(merged_list)
    merged_list = merged_list[:1000]
    train_data = merged_list[:int(0.7 * len(merged_list))]
    val_data = merged_list[int(0.7 * len(merged_list)):]

    def fetch_data(data):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.shuffle(buffer_size=200)
        dataset = dataset.map(lambda x: ({"image_input": tf.image.per_image_standardization(imread(x[0], 3))},
                                         {"labels": imread(x[1], 1)}),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(lambda x, y: (x, {"labels": onehot_labels(y["labels"])}))
        dataset = dataset.batch(8)
        return dataset.prefetch(buffer_size=200)

    return fetch_data(train_data), fetch_data(val_data)
