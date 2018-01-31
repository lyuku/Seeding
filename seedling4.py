import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.applications import xception
from keras.layers import Dense, GlobalAveragePooling2D,Activation
from keras.models import Model,Sequential,Input

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



data_dir = './input/plant-seedlings-classification/'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
output = './output/out'

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
        img = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
        x[i] = img

    return x, y

train_x, train_y = process_data(train_dir)

val_x, val_y = process_data(val_dir)


train_x_bf = base_model.predict(train_x, batch_size=32, verbose=1)

val_x_bf = base_model.predict(val_x, batch_size=32, verbose=1)


def one_hot(data):
    ont_hot = tf.one_hot(data, depth=12, axis=-1)
    sess = tf.Session()
    hot_data = sess.run(ont_hot)
    sess.close()
    return hot_data

train_y_hot = one_hot(train_y)
val_y_hot = one_hot(val_y)

print(train_x_bf.shape)
print(train_y_hot.shape)

# logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=1987)
# logreg.fit(train_x_bf, train_y)
#
#
# valid_probs = logreg.predict_proba(val_x_bf)
# valid_preds = logreg.predict(val_x_bf)
#
# print('Validation Xception Accuracy {}'.format(accuracy_score(val_y, valid_preds)))

# model = Sequential()
# model.add(Dense(2048, activation="relu", input_shape=(2048, )))
# model.add(Dense(12, activation="softmax"))

model = Sequential()
model.add(Dense(2048, input_shape=(2048, )))
model.add(Activation("relu"))
model.add(Dense(12))
model.add(Activation("softmax"))

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(x=train_x_bf, y=train_y_hot, batch_size=32, epochs=10)

model.save(output)

# valid_probs = model.predict_proba(val_x_bf)
#
# valid_preds = model.predict(val_x_bf)
#
# print(valid_preds.shape)
# print(val_y_hot.shape)
#
# print('Validation Xception Accuracy {}'.format(accuracy_score(val_y_hot, valid_preds)))

pred = model.evaluate(val_x_bf, val_y_hot)

print("Loss = " + str(pred[0]))
print("Accuracy = " + str(pred[1]))


def new_last_layer(base_model, nb_class):

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(nb_class, activation='softmax')(x)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def setup_to_transfer_learn(model):

    for layer in model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# model = new_last_layer(base_model, 12)
#
# setup_to_transfer_learn(model)
#
# model.save(output)