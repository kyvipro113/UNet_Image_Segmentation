from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

json_model_save = open("model_save.json", "r")
load_model_json = json_model_save.read()
json_model_save.close()
model = model_from_json(load_model_json)
model.load_weights("weights_unet.hdf5")

img_path = "data/membrane/test/0.png"
img_test = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#cv2.imshow("Image Test", img_test)
#cv2.waitkey()

ax1.imshow(img_test)
ax1.set_title("Image Test")
ax1.legend()

img_test = img_test / 255
img_test = cv2.resize(img_test, (256, 256))

img_test = np.expand_dims(img_test, axis=2)
img_test = np.expand_dims(img_test, axis=0)
result = model.predict(img_test, verbose=0)
#print(result[0].shape)
#cv2.imshow("Image Test Predict", result[0]*255)
#cv2.waitkey()

ax2.imshow(result[0]*255)
ax2.set_title("Image Test Predict")
ax2.legend()

plt.show()
