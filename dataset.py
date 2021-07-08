from glob import glob

import cv2
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


def imread(file_path, channels):
    return tf.reshape(tf.image.resize_with_pad(image=tf.image.decode_image(tf.io.read_file(file_path), channels=channels),
                                               target_width=224, target_height=224), shape=[224, 224, channels])


def onehot_labels(input_tensor):
    input_tensor = tf.cast(tf.greater_equal(tf.squeeze(input_tensor, axis=-1), 0.001), tf.dtypes.uint8)
    return tf.keras.backend.one_hot(indices=input_tensor, num_classes=2)


def augm(image, label):
    # random brightness adjustment illumination
    image["image_input"] = tf.image.random_brightness(image["image_input"], 0.3)
    # random contrast adjustment
    image["image_input"] = tf.image.random_contrast(image["image_input"], 0.2, 0.5)

    if np.random.uniform() > 0.5:
        image["image_input"] = tf.image.flip_left_right(image["image_input"])
        label["labels"] = tf.image.flip_left_right(label["labels"])
    if np.random.uniform() > 0.5:
        image["image_input"] = tf.image.flip_up_down(image["image_input"])
        label["labels"] = tf.image.flip_up_down(label["labels"])

    rot_factor = tf.cast(tf.random.uniform(shape=[], maxval=12, dtype=tf.int32), tf.float32)
    angle = np.pi / 12 * rot_factor
    image["image_input"] = tfa.image.rotate(image["image_input"], angle, interpolation="nearest")
    label["labels"] = tfa.image.rotate(label["labels"], angle, interpolation="nearest")
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
    # merged_list = merged_list[:1000]
    train_data = merged_list[:int(0.7 * len(merged_list))]
    val_data = merged_list[int(0.7 * len(merged_list)):]

    def fetch_data(data, mode="train"):
        assert mode in ("train", "val")
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.shuffle(buffer_size=200)
        dataset = dataset.map(lambda x: ({"image_input": imread(x[0], 3)},
                                         {"labels": imread(x[1], 1)}),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if mode == "train":
            dataset = dataset.map(augm, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(lambda x, y: ({"image_input": tf.image.per_image_standardization(x["image_input"])},
                                            {"labels": onehot_labels(y["labels"])}))
        dataset = dataset.batch(8)
        return dataset.prefetch(buffer_size=200)

    return fetch_data(train_data, "train"), fetch_data(val_data, "val")


if __name__ == '__main__':
    cv2.namedWindow("image", cv2.WINDOW_FREERATIO)
    train, val = get_datasets()
    cv2.namedWindow("label", cv2.WINDOW_FREERATIO)

    for image, label in train.as_numpy_iterator():
        image_slice = image["image_input"][0, :, :, :]
        image_slice = np.divide(image_slice - np.min(image_slice),
                                np.max(image_slice) - np.min(image_slice))
        image_slice = image_slice * 255
        image_slice = image_slice.astype(np.uint8)
        cv2.imshow("image", image_slice)  # np.divide(image_slice, np.max(image_slice)))
        cv2.imshow("label", label["labels"][0, :, :, 1])

        cv2.waitKey()
