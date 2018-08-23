import os
import numpy as np
from keras.preprocessing import image
from keras.applications import xception

import matplotlib.pyplot as plt

import data_process


data_dir = './input/plant-seedlings-classification/'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
output = './output/out'

train_out = './output/train_out'
val_out = './output/val_out'

preprocess_input = xception.preprocess_input
batch_size = 64
IM_WIDTH = 299
IM_HEIGHT = 299
FC_SIZE = 1024


base_model = xception.Xception(include_top=False, weights="imagenet", pooling="avg")

categories = os.listdir(train_dir)

def read_img(filepath, size):
    img = image.load_img(filepath, target_size=size)
    img = image.img_to_array(img)
    return img

def process_data(path):
    files = []
    y = []
    for (label, name) in ((i+1, categories[i]) for i in range(0, 12)):
        dir = os.path.join(path, name)
        for file in os.listdir(dir):
            files.append(os.path.join(dir, file))
            y.append(label)

    x = np.zeros((len(files), IM_WIDTH, IM_HEIGHT, 3), dtype="float32")

    for i in range(len(files)):
        img = read_img(files[i], (IM_WIDTH, IM_HEIGHT))
        img = data_process.segment_plant(img)
        img = data_process.sharpen_image(img)
        img = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
        x[i] = img

    return x, y


# img = read_img('./input/plant-seedlings-classification/test/0ad9e7dfb.png', (IM_WIDTH, IM_HEIGHT))
#
# img1 = data_process.segment_plant(img)
#
# img2 = data_process.sharpen_image(img1)
#
# fig, axs = plt.subplots(1, 3, figsize=(20, 20))
#
# axs[0].imshow(img)
# axs[1].imshow(img1)
# axs[2].imshow(img2)
#
# plt.show()

train_x, train_y = process_data(train_dir)

val_x, val_y = process_data(val_dir)

train_x_bf = base_model.predict(train_x, batch_size=32, verbose=1)

val_x_bf = base_model.predict(val_x, batch_size=32, verbose=1)

np.save(train_out, train_x_bf)
np.save(val_out, val_x_bf)
#
#
# def one_hot(data):
#     ont_hot = tf.one_hot(data, depth=12, axis=-1)
#     sess = tf.Session()
#     hot_data = sess.run(ont_hot)
#     sess.close()
#     return hot_data
#
# train_y_hot = one_hot(train_y)
# val_y_hot = one_hot(val_y)
#
# model = load_model(output)
#
# history = model.fit(x=train_x_bf, y=train_y_hot, batch_size=32, epochs=2)
#
# model.save(output)
#
# # model = load_model(output)
#
# pred = model.evaluate(val_x_bf, val_y_hot)
#
# print(history.history)
#
# print("Loss = " + str(pred[0]))
# print("Accuracy = " + str(pred[1]))
