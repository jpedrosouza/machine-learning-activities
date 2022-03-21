import keras
import cv2 as cv
import urllib.request
from PIL import Image, ImageOps
import numpy as np

np.set_printoptions(suppress=True)

model = keras.models.load_model('./models/transfer-learning-cat-dog-identifier/transfer-learning-cat-dog-identifier.h5')

data = np.ndarray(shape=(1, 160, 160, 3), dtype=np.float32)

urllib.request.urlretrieve(
  'https://static1.patasdacasa.com.br/articles/6/82/6/@/16964-a-lingua-do-gato-tem-funcoes-desde-a-apr-opengraph_1200-2.jpg',
   "./sample/animal.png")

image = Image.open('./sample/animal.png')
size = (160, 160)
image = ImageOps.fit(image, size, Image.ANTIALIAS)
image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

data[0] = normalized_image_array

prediction = model.predict(data)

print('')

print(prediction)

print('')

if (prediction[0][0] * -1 > 0.9):
    print('Gato')
else:
    print('Cachorro')
