import numpy as np
import os
import pandas
from PIL import Image
from tensorflow.keras.models import load_model


def scenario_gray(image): #grayscale picture
    return image.convert('LA')

#definitions
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

#Create 4D array for, [case representing fog lv][sign][picture][predictions(sorted)]
# 18,2,41
predictions_1 = np.empty((number_of_test_signs_categories,number_of_test_files_in_dir,number_of_all_prediction_categories), dtype=object)
predictions_2 = np.empty((number_of_test_signs_categories,number_of_test_files_in_dir,number_of_all_prediction_categories), dtype=object)
predictions_3 = np.empty((number_of_test_signs_categories,number_of_test_files_in_dir,number_of_all_prediction_categories), dtype=object)
predictions_4 = np.empty((number_of_test_signs_categories,number_of_test_files_in_dir,number_of_all_prediction_categories), dtype=object)

predictions = [predictions_1, predictions_2, predictions_3, predictions_4]

# load test images
for i in range(0,4):
  # j_num = 0
  for j,j_num in zip(signs_indexes_in_test_dataset, range(0,len(signs_indexes_in_test_dataset))):

    path = os.path.join(os.getcwd(), 'test_sets' +"\\"+str(j)+ '\\'  + 'fog',str(i+1) )
    images = os.listdir(path)
    # print(images)
    for k,k_num in zip(images, range(0,number_of_test_files_in_dir)):
      try:

        image = Image.open(path + '\\' + k)
        image = image.resize((30, 30))
        image = scenario_gray(image)
        image = np.array(image)
        data_test.append(image)
        labels_test.append(i)

        prediction_distribution =model.predict(np.array([image]))

        for l in range(0,40):
          # print(i)# print(j_num)# print(k_num)# print(l)# print("\n")
          predictions[i][j_num][k_num][l] = prediction_distribution[0][l]

        #k_num = +1
      except:
        print("Error loading test image")

  # print("final prediction table")
  # print(predictions[i][j_num])


# Converting lists into numpy arrays
data = np.array(data_test)
labels = np.array(labels_test)

x = predictions

#Read classifiers ids form label to list
list_index = df.ClassId.to_list()
list_index.remove("ClassId")
#Read classifiers names form label to list
list_clasification_names = df.Name.to_list()
list_clasification_names.remove("Name")
# Sort the predictions

verdict = np.empty((number_of_test_cases,number_of_test_signs_categories,number_of_test_files_in_dir), dtype=object)
# print("verdict shape ")
# print(np.shape(verdict))
# print("classifiers id list len")
# print(len(list_index))

for i in range(0,number_of_test_cases):
  for j in range(0, number_of_test_signs_categories):
    for k in range(0, number_of_test_files_in_dir):
      list_index_x = list_index
      for l in range(len(list_index_x )-1 ):# -1
        for m in range(len(list_index_x)-1 ): # -1
          if x[i][j][k][int(list_index[l])] > x[i][j][k][int(list_index[m])]:
            temp = list_index_x[l]
            list_index_x[l] = list_index_x[m]
            list_index_x[m] = temp

      # print(list_index_x)
      # print("first index: ",int(list_index_x[0]), "\n")
      verdict[i][j][k] = list_index_x[0]


#Show the sorted labels in order from highest probability to lowest
# print(list_index)
for i in range(0,number_of_test_cases):
  for j in range(0, number_of_test_signs_categories):
    for k in range(0, number_of_test_files_in_dir):
      print("fog level: " + str(i + 1))
      print("should be: ", signs_indexes_in_test_dataset[j])
      # for n in range(1):
      print(" verdict ",verdict[i][j][k]," ",list_clasification_names[int(verdict[i][j][k])], ':', round(predictions[i][j][k][int(verdict[i][j][k])] * 100, 2), '%')
      print("\n")

#TODO: case for insert just one picture and get prediction (for GUI)

#path from received from gui
#after join with GUI this method should take one param - path (received from GUI)
def predictImage():
  print("RUN PREDICT IMAGE")
  path = os.path.join(os.getcwd(), 'test_sets' + "\\" + str(0) + '\\' + 'fog', str(1))
  images = os.listdir(path)
  image = Image.open(path + '\\' + "2019_0719_142601_003 1954_0.jpg")
  #image = Image.open(path)
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

#uncomment to test method
#predictImage()