import cv2
import numpy as np
import tensorflow as tf
from model import dice_coef_loss

model = tf.keras.models.load_model('modell',custom_objects={'dice_coef_loss': dice_coef_loss})

image = cv2.resize(cv2.imread("/home/livanosg/projects/mel-seg/data/ph2/data/IMD018/IMD018_Dermoscopic_Image/IMD018.bmp"), (224, 224))
output = model.predict(tf.expand_dims(image, axis=0))
output = output[0, :, :, 0]
cv2.namedWindow("Input", flags=cv2.WINDOW_FREERATIO)
cv2.namedWindow("Results", flags=cv2.WINDOW_FREERATIO)
cv2.imshow("Input", image)
cv2.imshow("Results", output)
cv2.waitKey()
