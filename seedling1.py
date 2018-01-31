import os
from keras.preprocessing.image import ImageDataGenerator

from keras.applications import xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

data_dir = './input/plant-seedlings-classification/'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
output = './output/out'

preprocess_input = xception.preprocess_input
batch_size = 64
IM_WIDTH = 299
IM_HEIGHT = 299
FC_SIZE = 1024

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
)
validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
)

base_model = xception.Xception(include_top=False, weights="imagenet")

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

model = new_last_layer(base_model, 12)

setup_to_transfer_learn(model)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=1000,
    epochs=2,
    validation_data=validation_generator,
    validation_steps=200,
    class_weight='auto')

model.save(output)