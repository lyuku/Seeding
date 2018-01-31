import os
import shutil

data_dir = './input/plant-seedlings-classification/'
train_dir = os.path.join(data_dir, 'train/')
val_dir = os.path.join(data_dir, 'val/')

for dir in os.listdir(train_dir):
    fulldir = train_dir + dir + "/"
    valdir = val_dir + dir + "/"
    os.mkdir(valdir)
    i = 0
    for file in os.listdir(fulldir):
        if i < 40:
           shutil.move(fulldir+file, valdir+file)
           i += 1
