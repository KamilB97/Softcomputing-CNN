from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas
from PIL import Image
from tensorflow.keras.models import load_model


############################## DEFINITIONS
model_name = "my_model.h5"
label_file = "labels.csv"
label_col_names = ["ClassId", "Name"]
path_test = "test_sets"
data_test = []
labels_test = []
signs_indexes_in_test_dataset = [0,1,2,3,4,11,12,13,14,20,23,24,26,30,31,32,33,36]
number_of_test_cases = 4
number_of_test_signs_categories = len(signs_indexes_in_test_dataset)
number_of_test_files_in_dir = 2
number_of_all_prediction_categories = 40
#Images display
columns = 2
rows = 1
#load csv columns data
df = pandas.read_csv(label_file, names=label_col_names)

# load model
model = load_model(model_name)

def scenario_gray(image): #grayscale picture
    return image.convert('LA')

#path from received from gui
#after join with GUI this method should take one param - path (received from GUI)
def predictImage(path):
  print("RUN PREDICT IMAGE")
  #Uncomment to insert image path manually
  # path = os.path.join(os.getcwd(), 'test_sets' + "\\" + str(0) + '\\' + 'fog', str(1))
  # images = os.listdir(path)
  # image = Image.open(path + '\\' + "2019_0719_142601_003 1954_0.jpg")
  image = Image.open(path)
  image = image.resize((30, 30))
  image = scenario_gray(image)
  image = np.array(image)
  data_test.append(image)
  labels_test.append(0)

  prediction_distribution = model.predict(np.array([image]))

  x = prediction_distribution

  # Read classifiers ids form label to list
  list_index = df.ClassId.to_list()
  list_index.remove("ClassId")
  # Read classifiers names form label to list
  list_clasification_names = df.Name.to_list()
  list_clasification_names.remove("Name")
  # Sort the predictions


  for i in range(len(list_index )-1):
    for j in range(len(list_index )-1):
      if x[0][int(list_index[i])] > x[0][int(list_index[j])]:
        temp = list_index[i]
        list_index[i] = list_index[j]
        list_index[j] = temp
  # Show the sorted labels in order from highest probability to lowest
  # print(list_index)

#Show 5 highest probabilities
  for i in range(5):
    print(list_clasification_names[int(list_index[i])], ':', round(prediction_distribution[0][int(list_index[i])] * 100, 2), '%')

#predictImage()

def openfn():

    filename = filedialog.askopenfilename(title='open')
    return filename

def open_img():
    # Select the Imagename  from a folder
    # img is retrived image
    x = openfn()
    print(x)
    img = Image.open(x)
    img = img.resize((300, 300), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.grid(row=2)
    return img


root = Tk()
root.title("Image Loader")
root.geometry("850x600+300+150")
root.resizable(width=True, height=True)
btn = Button(root, text='Load image', command = open_img).grid(row=1, columnspan=4)


root.mainloop()



