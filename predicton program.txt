import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
import os
import csv
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn import datasets
from tkinter import ttk
from matplotlib.ticker import FuncFormatter
from tkinter import *
book = openpyxl.load_workbook("C:\\random forest algorithm\\superstore.xlsx")
sheetR = book['sheet1']
l=[]
l2=[]
l4=[]
for r in range (sheetR.max_row-1):
    cell1 = sheetR.cell(row=r+2,column=26).value
    l.append(cell1)
h=0.70*sheetR.max_row
for r in range (int(h),sheetR.max_row-1):
    cell1 = sheetR.cell(row=r+2,column=14).value
    l4.append(cell1)

for i in range(int(h)-1,sheetR.max_row-1):
    if l[i]==101:
        l2.append("Environmental")
    if l[i]==102:
        l2.append("Individual")
    if l[i]==103:
        l2.append("Organisational")
    if l[i]==104:
        l2.append("Interpersonal")

AH_data = pd.read_excel("C:\\random forest algorithm\\superstore.xlsx")
data_clean = AH_data.dropna()
data_clean.head()
data_clean.dtypes # data types of each variable
data_clean.describe()
predictors = data_clean[['cust_segment','product_cat','Order Quantity','Sales','region','Unit Price']]
#print predictors
targets = data_clean.behaviour
pred_train, pred_test,tar_train, tar_test = train_test_split(predictors,targets, test_size=0.3,shuffle=False)


# shape/dimensions of the DataFrame
pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape
# Build model on training data
from sklearn.ensemble import RandomForestClassifier
# n_estimators is the amount of trees to build
classifier=RandomForestClassifier(n_estimators=7)
# fit the RandomForest Model
classifier=classifier.fit(pred_train,tar_train)
predictions=classifier.predict(pred_test)
l3=[]

for i in range(len(predictions)):
  #  print (predictions[i])
    if predictions[i]==101:
        l3.append("Environmental")
    if predictions[i]==102:
        l3.append("Individual")
    if predictions[i]==103:
        l3.append("Organisational")
    if predictions[i]==104:
        l3.append("Interpersonal")
        
print ("Predicted Behaviour",'\t',"Actual Behaviour")
print ("-----------------------------------")
for i in range(1,47):
    print (l3[i],'\t\t',l2[i])
print ("-----------------------------------")
# confusion matrix / missclassification matrix
print ("confusion metrics:\n",sklearn.metrics.confusion_matrix(tar_test,predictions))
print ("-------------------")
print ('accuracy :',sklearn.metrics.accuracy_score(tar_test, predictions))
print ("-------------------")
print ("Report :\n",sklearn.metrics.classification_report(tar_test,predictions))
print ("-------------------")


l7=[]
l8=[]
l9=[]
l10=[]
for i in range(1,80):
    if predictions[i]==101:
        l7.append("Environmental")
    if predictions[i]==102:
        l8.append("Individual")
    if predictions[i]==103:
        l9.append("Organisational")
    if predictions[i]==104:
        l10.append("Interpersonal")
#for i in range(1,47):
    x=len(l7)
    y=len(l8)
    z=len(l9)
    r=len(l10)
# creating the dataset
genres=['Environmental', 'Individual', 'Organisation','Interpersonal']
data = {x,y,z,r}

l11=[]
l12=[]
l13=[]
l14=[]
for i in range(1,80):
    if l[i]==101:
        l11.append("Environmental")
    if l[i]==102:
        l12.append("Individual")
    if l[i]==103:
        l13.append("Organisational")
    if l[i]==104:
        l14.append("Interpersonal")
        
    f=len(l11)
    g=len(l12)
    h=len(l13)
    i=len(l14)
# creating the dataset
data1 = {f,g,h,i}

values=np.arange(len(genres))
width=0.4
def plot(data1,values,width):
 barwidth=0.35
 plt.bar(values,data1,color='royalblue',width=barwidth,label='ff')
 #plt.bar(values+width,data1,width)
 plt.xlabel("Behaviour")
 plt.ylabel("Frequency")
 plt.title("prediction")
 plt.xticks(values,genres)
 barwidth=0.2
 
 plt.legend("Behaviour")
 plt.show()

def openfile():
    file=filedialog.askopenfilename(title="Open a File", filetype=(("xlxs files", ".*xlsx"),
("All Files", "*.")))
    if file:
        try:
         file=r"{}".format(file)
         df=pd.read_excel(file)
        except ValueError:
            label.config(text="file could not be open")
        except FileNotFoundError:
            label.config(text="file not found")
    clear_treeview()
    # Add new data in Treeview widget
    tree["column"] = list(df.columns)
    tree["show"] = "headings"

   # For Headings iterate over the columns
    for col in tree["column"]:
      tree.heading(col, text=col)

   # Put Data in Rows
    df_rows = df.to_numpy().tolist()
    for row in df_rows:
         tree.insert("", "end", values=row)

    tree.place(x=10,y=230)

# Clear the Treeview Widget
def clear_treeview():
   tree.delete(*tree.get_children())

def n():
 global t
 global tree
 t=Tk()
 Label(t,text = "PREDICTION OF CONSUMER BEHAVIOUR", height = 4,fg = "white",font=(120),bg="pink",width=143).place(x=0,y=0)

 tree = ttk.Treeview(t)
 Button(t,text="PLOT",command=lambda:plot(data1,values,width)).place(x=10,y=480)

 t.mainloop()
n()