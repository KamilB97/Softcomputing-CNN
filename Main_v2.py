# https://www.lfd.uci.edu/~gohlke/pythonlibs/ site with libs in the case when pycharm cannot find appropriate version

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

################## METHODS ##################
def scenario_gray(image): #grayscale picture
    return image.convert('LA')
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

#Images display
columns = 2
rows = 1

for i in range(classes):
    path = os.path.join(os.getcwd(), 'datasets', str(i))
    
    images = os.listdir(path)
    showFirst = True
    for j in images:
        try:
            ######## SCENARIOS ########
            image = Image.open(path + '/' + j)
            image = image.resize((30, 30))
            image = scenario_gray(image)      # uncomment to scenario 1

            # threshold = 75                                      # uncomment to scenario 2
            # image = scenario_black_white_normal(image, threshold) # uncomment to scenario 2

            #threshold = 75                                         # uncomment to scenario 3
            #image = scenario_black_white_granular(image, threshold)  # uncomment to scenario 3


            # counter += 1    # if you want more picture samples (more windows with pictures), decrease it. If want less samples, increase
            # frequency = 800 # if you want different set of immages just change a little this value          
                                           
            if showFirst:
                fig = plt.figure(figsize=(8, 8))
                 # Print pictures to screen. Close window to run next pair of images
                fig.add_subplot(rows, columns, 1) # first image is the processed image
                plt.imshow(image)
                fig.add_subplot(rows, columns, 2) # second image is the original image
                plt.imshow(Image.open(path + '/' + j))
                showFirst=False
                
            image = np.array(image)
            data.append(image)
            labels.append(i)

        except:
            print("Error loading image")

#Display images
plt.show()

# Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)
print(data.shape, labels.shape)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=68)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#Converting the labels into one hot encoding
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

#Building the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(classes, activation='softmax'))

#Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 10
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
model.save("my_model.h5")

#plotting graphs for accuracy 
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()