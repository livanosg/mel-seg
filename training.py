from datetime import datetime

import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
# import tensorflow_addons as tfa
from callbacks import CyclicLR
from metrics import f1
from model import unet
from dataset import get_datasets
from losses import log_dice_loss

callbacks = [tf.keras.callbacks.TensorBoard(log_dir=f"logs/trial_{datetime.now().strftime('%d%m%y%H%M%S')}"),
             CyclicLR(base_lr=0.0001, max_lr=0.0001 * 5, mode='exp_range', gamma=0.999),
             EarlyStopping(verbose=1, patience=10),
             ModelCheckpoint(filepath=f"model_{datetime.now().strftime('%d%m%y%H%M%S')}", save_best_only=True)]
train_data, val_data = get_datasets()
model = unet()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
              loss=log_dice_loss,
              metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC(),
                       tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                       tf.keras.metrics.MeanIoU(num_classes=2), f1
                       # tfa.metrics.F1Score(num_classes=2),  # 'micro' == accuracy
                       ])

model.fit(train_data, validation_data=val_data, callbacks=callbacks, epochs=100)
model.save(f"model_{datetime.now().strftime('%d%m%y%H%M%S')}")
