import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
import random

from sklearn.model_selection import train_test_split


# https://www.lfd.uci.edu/~gohlke/pythonlibs/
path = "datasets"
labelFile = 'labels.csv'

epochs_val=10
imageDimensions=(30,30,3)
testRatio=0.2
validationRatio=0.2

count=0
images=[]
classNo=[]
myList = os.listdir(path)
print("Classes detected: ",len(myList))
noOfClasses=len(myList)
print("importing classes........")
for x in range(0,len(myList)):
    myPicList = os.listdir(path +"/"+str(count))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(count)+"/"+y)
        images.append(curImg)
        classNo.append(count)  
    print(count, end =" ")
    count +=1
print(" ")
images = np.array(images)
classNo = np.array(classNo)

x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validationRatio)

#x_train is array of images to train and y_train is corresponding class id

print("data shapes")
print("Train",end =""); print(x_train.shape, y_train.shape)
print("Validation",end=""); print(x_validation.shape,y_validation.shape)
print("test", end=""); print(x_test.shape, y_test.shape)

#assert(x_train.shape[0]==y_train.shape[0]), "train number of images is not equal to the number of labels in training set"
#assert(x_validation.shape[0]==y_validation.shape[0]), "the number of images is not eqyaual to the number of labels in validation set"
#assert(x_test.shape==y_test.shape[0]), "the number of images is not equal to the number of labels in test set"
#assert(x_train.shape[1:]==(imageDimensions)), "the dimensions of the training images are wrong"
#assert(x_validation.shape[1:]==(imageDimensions)),"the dimesions of the validation images are wrong"
#assert(x_test.shape[1:]==(imageDimensions)), "the dimensions of the test images are wrong"

data= pd.read_csv(labelFile, error_bad_lines=False)
print("data shape"), data.shape, type(data)

num_of_samples = []
cols=5
num_classes = noOfClasses
fig,axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5,300))
fig.tight_layout()
for i in range(cols):
    for j,row in data.iterrows():
        x_selected  = x_train[y_train ==j]
        axs[j][i].imshow(x_selected[random.randint(0,len(x_selected)-1),:,:], cmap=plt.get_camp("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j)+ "-"+row["Name"])
            num_of_samples.append(len(x_selected))


#samples for each categoriy

print(num_of_samples)
plt.figure(figsize=(12,4))
plt.bat(range(0,num_classes), num_of_samples)
plt.title("distribution of the training dataset")
plt.xlabel("class number")
plt.ylabel("number of images")
plt.show

def grayscale(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img=cv2.equalizeHist(img)
    return img
def imagePreprocessing(img):
    img=grayscale(img)
    img=equalize(img)
    img=img/255 #normalize vlues betwween 0 an 1

x_train=np.array(list(map(imagePreprocessing,x_train))) #preprocess all images
x_test=np.array(list(map(imagePreprocessing,x_test)))
x_validation=np.array(list(map(imagePreprocessing,x_validation)))

cv2.imshow("Gray image", x_train[random.randint(0, len(x_train)-1)]) #validation if preprocessing is correctly done