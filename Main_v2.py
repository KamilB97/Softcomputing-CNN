#import csv
#import pickle
#import cv2
#import pandas as pd
#import random
#import time
#import tqdm
# from keras.utils import to_categorical
# from keras.models import Sequential, load_model
# from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
# https://www.lfd.uci.edu/~gohlke/pythonlibs/ site with libs in the case when pycharm cannot find appropriate version

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

################## METHODS ##################
def scenario_gray(image): #grayscale picture
    image = ImageOps.grayscale(image)
    return image
def scenario_black_white_normal(image, treshold):
    fn = lambda x: 255 if x > treshold else 0
    image = image.convert('L').point(fn, mode='1')
    return image
def scenario_black_white_granular(image, treshold):
    image = image.convert('1') # morre granuar image - better readability of signs in my opinion
    return image

#################################################
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
            ######## SCENARIOS ########
            image = Image.open(path + '\\' + j)
            image = ImageOps.grayscale(image) # uncomment to scenario 1
            image = scenario_gray(image)      # uncomment to scenario 1

            # treshold = 75                                      # uncomment to scenario 2
            # image = scenario_black_white_normal(image, thresh) # uncomment to scenario 2

            # treshold = 75                                         # uncomment to scenario 3
            # image = scenario_black_white_granular(image, thresh)  # uncomment to scenario 3

            image = image.resize((30, 30))


            counter += 1
            frequency = 400                # if you want more picture samples (more windows with pictures), decrease it. If want less samples, increase
                                           # if you want different set of immages just change a little this value
            if counter%frequency ==0:
                print("j value: " + j)
                w = 10
                h = 10
                fig = plt.figure(figsize=(8, 8))
                columns = 2
                rows = 1

                # Print pictures to screen. Close window to run next pair of images
                fig.add_subplot(rows, columns, 1) # first image is the processed image
                plt.imshow(image)
                fig.add_subplot(rows, columns, 2) # second image is the original image
                plt.imshow(Image.open(path + '\\' + j))
                plt.show()

            image = np.array(image)
            data.append(image)
            labels.append(i)

        except:
            print("Error loading image")


# Converting lists into numpy arrays

data = np.array(data)
labels = np.array(labels)
print(data.shape, labels.shape)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=68)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


