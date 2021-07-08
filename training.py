from datetime import datetime

import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
# import tensorflow_addons as tfa
from metrics import f1
from model import unet
from dataset import get_datasets
from losses import log_dice_loss, custom_loss

callbacks = [tf.keras.callbacks.TensorBoard(log_dir=f"logs/trial_{datetime.now().strftime('%d%m%y%H%M%S')}"),
             # CyclicLR(base_lr=0.0001, max_lr=0.0001 * 5, mode='exp_range', gamma=0.999),
             EarlyStopping(verbose=1, patience=10),
             ModelCheckpoint(filepath=f"model_{datetime.now().strftime('%d%m%y%H%M%S')}", save_best_only=True)]
train_data, val_data = get_datasets()
model = unet()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
              loss=custom_loss,
              metrics=[tf.keras.metrics.CategoricalAccuracy(),
                       tf.keras.metrics.AUC(),
                       f1,
                       # tfa.metrics.F1Score(num_classes=2, average='micro'),  # 'micro' == accuracy
                       ])

model.fit(train_data, validation_data=val_data, callbacks=callbacks, epochs=100)
