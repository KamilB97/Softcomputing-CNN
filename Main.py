import csv

import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
import random
import time

from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import tqdm



# https://www.lfd.uci.edu/~gohlke/pythonlibs/
path = "datasets"
labelFile = 'labels.csv'

data = []
labels = []
classes = 40

filename = "labels.csv"

# load dictionary
# with open(filename, 'r') as data:
#     for line in csv.reader(data):
#         print(line)

data = []
labels = []
classes = 40

for i in range(classes):
    path = os.path.join(os.getcwd(), 'datasets', str(i))
    images = os.listdir(path)
    counter = 0;
    for j in images:
        try:
            image = Image.open(path + '\\' + j)
            image = ImageOps.grayscale(image)
            image = image.resize((30, 30))
            # showImageTest, delay 5 sec, brief
            # counter += 1
            # if counter%400 ==0:
                # print("j value: " + j)
                #
                # image.show()
                # time.sleep(5)

            image = np.array(image)
            data.append(image)
            labels.append(i)

        except:
            print("Error loading image")
# Converting lists into numpy arrays
#
data = np.array(data)
labels = np.array(labels)
print(data.shape, labels.shape)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=68)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


