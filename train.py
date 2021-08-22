from tensorflow.python.training.tracking.util import Checkpoint
from unet import *
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import *
import pandas as pd
import os
import numpy as np
import cv2
import skimage.io as io
import skimage.transform as trans

train_path = "data/membrane/train/"
test_path = "data/membrane/test/"

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANELS = 1

total_image_train = len(os.listdir(train_path + "image")) - 1 # If windows have not thumbs.db remove -1
total_mask_train = len(os.listdir(train_path + "mask")) - 1 # If windows have not thumbs.db remove -1
print(total_image_train)
print(total_mask_train)


X_train = np.zeros((total_image_train, IMG_WIDTH, IMG_HEIGHT, IMG_CHANELS), dtype=np.float)
Y_train = np.zeros((total_image_train, IMG_WIDTH, IMG_HEIGHT, IMG_CHANELS), dtype=np.float)


i = 0
for img_file in os.listdir(train_path + "image"):
    if os.path.splitext(img_file)[1] in [".jpg", ".png"]:
        img = io.imread(train_path + "image/" + img_file)
        img = img /255
        img = np.expand_dims(trans.resize(img, (IMG_HEIGHT, IMG_WIDTH), mode="constant", preserve_range=True), axis=-1)
        X_train[i] = img
        i += 1
    else:
        pass

i = 0
for mask_file in os.listdir(train_path + "mask"):
    if os.path.splitext(mask_file)[1] in [".jpg", ".png"]:
        mask_ = io.imread(train_path + "mask/" + mask_file)
        mask_ = mask_/255
        mask_ = np.expand_dims(trans.resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode="constant", preserve_range=True), axis=-1)
        Y_train[i] = mask_
        i += 1
    else:
        pass

#print(X_train)
#print(Y_train)

INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANELS)

model = unet(input_shape=INPUT_SHAPE)

BATCH_SIZE = 3
STEPS_PER_EPOCH = 3
EPOCHS = 10

check_point = ModelCheckpoint('weights_unet.hdf5', monitor='loss',verbose=1, save_best_only=True)

history = model.fit(x=X_train, y=Y_train, validation_split=0.1, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[check_point])

epochs = [i for i in range(1, EPOCHS + 1)]

loss = history.history["loss"]
acc = history.history["accuracy"]

val_loss = history.history["val_loss"]
val_acc = history.history["val_accuracy"]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

ax1.plot(epochs, loss, label="Loss Training", color="green")
ax1.plot(epochs, val_loss, label="Loss Validation", color="blue")
ax1.set_title("Training & Validation Loss")
ax1.legend()

ax2.plot(epochs, acc, label="Accurancy Training", color="green")
ax2.plot(epochs, val_acc, label="Accuracy Validation", color="blue")
ax2.set_title("Training & Validation Accuracy")
ax2.legend()

plt.show()

pd.DataFrame.from_dict(history.history).to_csv("history.csv", index=False)
history_data = pd.read_csv("history.csv", sep=",")
history_data.head()
print(history_data)

model_json_save = model.to_json()
with open ("model_save.json", "w") as json_save_file:
  json_save_file.write(model_json_save)
