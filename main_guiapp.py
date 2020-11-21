import numpy as np
import pandas
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image


############################## DEFINITIONS
model_name = "my_model.h5"
label_file = "labels.csv"
label_col_names = ["ClassId", "Name"]
path_test = "test_sets"
signs_indexes_in_test_dataset = [0,1,2,3,4,11,12,13,14,20,23,24,26,30,31,32,33,36]
number_of_test_cases = 4
number_of_test_signs_categories = len(signs_indexes_in_test_dataset)
number_of_test_files_in_dir = 2
number_of_all_prediction_categories = 40

#load csv columns data
df = pandas.read_csv(label_file, names=label_col_names)

# load model
model = load_model(model_name)

def scenario_gray(image): #grayscale picture
    return image.convert('LA')

#path from received from gui
def predictImage(path):

  image = Image.open(path)
  image = image.resize((30, 30))
  image = scenario_gray(image)
  image = np.array(image)

  prediction_distribution = model.predict(np.array([image]))

  x = prediction_distribution

  # Read classifiers ids form label to list
  list_index = df.ClassId.to_list()
  list_index.remove("ClassId")
  # Read classifiers names form label to list
  list_clasification_names = df.Name.to_list()
  list_clasification_names.remove("Name")
  # Sort the predictions order from highest probability to lowest

  for i in range(len(list_index )-1):
    for j in range(len(list_index )-1):
      if x[0][int(list_index[i])] > x[0][int(list_index[j])]:
        temp = list_index[i]
        list_index[i] = list_index[j]
        list_index[j] = temp

  #Show 5 highest probabilities
  for i in range(5):
    print(list_clasification_names[int(list_index[i])], ':', round(prediction_distribution[0][int(list_index[i])] * 100, 2), '%')

  return str(list_clasification_names[int(list_index[0])]) + ':' + str(round(prediction_distribution[0][int(list_index[0])] * 100, 2)) + '%' +'\n'+ str(list_clasification_names[int(list_index[1])]) + ':' + str(round(prediction_distribution[0][int(list_index[1])] * 100, 2)) + '%' + "\n"+ str(list_clasification_names[int(list_index[2])]) + ':' + str(round(prediction_distribution[0][int(list_index[2])] * 100, 2)) + '%'

#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Softcomputing Traffic sign classification')
top.configure(background='#e8e8e8')
label=Label(top,background='#e8e8e8', font=('arial',15,'bold'))
sign_image = Label(top)
def classify(file_path):
    global label_packed
    sign = predictImage(file_path)
    print(sign)
    label.configure(foreground='#303841', text=sign)
def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#495464', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass
upload=Button(top,text="Upload image",command=upload_image,padx=10,pady=5)
upload.configure(background='#495464', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Upload traffic sign to recognition",pady=20, font=('arial',20,'bold'))
heading.configure(background='#e8e8e8',foreground='#303841')
heading.pack()
top.mainloop()



