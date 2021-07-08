import cv2
import numpy as np
import tensorflow as tf
from losses import log_dice_loss
from metrics import f1


image = cv2.resize(cv2.imread("/home/livanosg/projects/mel-seg/data/ISIC-Archive/Data/Images/ISIC_0000007.jpeg"), (224, 224))

with tf.device('/cpu:0'):
    model = tf.keras.models.load_model('model_070721160221', custom_objects={'log_dice_loss': log_dice_loss,
                                                                             'f1': f1})
    output = model.predict(tf.image.per_image_standardization(tf.reshape(image, shape=[-1, 224, 224, 3])))
print(np.unique(output))
print(len(np.unique(output)))
output = np.asarray(output[0, :, :, 0] * 255./ np.max(output[0, :, :, 0]))
print(np.unique(output)[:10])
print(len(np.unique(output)))

print(output.shape)
# print(output)
# output[output < 0.99999999] = 0  # thresholding
# print(np.max(output))
cv2.namedWindow("Input", flags=cv2.WINDOW_FREERATIO)
cv2.namedWindow("Results", flags=cv2.WINDOW_FREERATIO)
cv2.imshow("Input", np.reshape(image, (224, 224, 3)))
cv2.imshow("Results", np.reshape(output, (224, 224, 1)))
cv2.waitKey()
exit()
