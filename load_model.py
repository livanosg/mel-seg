import cv2
import numpy as np
import tensorflow as tf
from losses import log_dice_loss
from metrics import f1

model = tf.keras.models.load_model('model_060721183636', custom_objects={'log_dice_loss': log_dice_loss,
                                                                         'f1': f1})

image = cv2.resize(
    cv2.imread("/home/livanosg/projects/mel-seg/data/ph2/data/IMD015/IMD015_Dermoscopic_Image/IMD015.bmp"), (224, 224))
output = model.predict(tf.expand_dims(image, axis=0))

output = output[0, :, :, 0]
output[output < 0.99999999] = 0  # thresholding
print(np.max(output))
cv2.namedWindow("Input", flags=cv2.WINDOW_FREERATIO)
cv2.namedWindow("Results", flags=cv2.WINDOW_FREERATIO)
cv2.imshow("Input", image)
cv2.imshow("Results", output)
cv2.waitKey()
exit()
