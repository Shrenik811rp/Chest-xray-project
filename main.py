import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


import cv2
import os
import random

dataset_folder = os.listdir("xray_dataset/chest_xray")

print(dataset_folder)
#print(np.__version__)

'''
Folder paths
'''
train_folder = "xray_dataset/chest_xray/train"

test_folder = "xray_dataset/chest_xray/test"

validation_folder ="xray_dataset/chest_xray/val"


'''
Test,Train,Val have these folders
'''
labels = ["NORMAL","PNEUMONIA"]

img_size = 50

def getData(dir):
    data = []

    for label in labels:

        path = os.path.join(dir,label)
        class_num = labels.index(label)

        for img in os.listdir(path):

            try:
                img_arr = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)

                new_arr = cv2.resize(img_arr,(img_size,img_size))

                data.append([new_arr,class_num])
            
            except Exception as error:
                print(error)
    
    return np.array(data,dtype=object)


train = getData(train_folder)

test = getData(test_folder)

val = getData(validation_folder)



'''
Splitting data into training and testing data

Extracting features and respective labels of data
'''
X_train = []
Y_train = []

X_val = []
Y_val = []

X_test = []
Y_test = []


'''

Appending features to X
Appending labels to Y

'''
for feature, label in train:
    X_train.append(feature)
    Y_train.append(label)

for feature, label in test:
    X_test.append(feature)
    Y_test.append(label)
    
for feature, label in val:
    X_val.append(feature)
    Y_val.append(label)


'''
Normalisation
'''

X_train = np.array(X_train) / 255

X_val = np.array(X_val) / 255

X_test = np.array(X_test) / 255


'''
Dimensions
'''
print(f"X_train shape: {X_train.shape}")

print(f"X_test shape: {X_test.shape}")

print(f"X_val shape: {X_val.shape}")

print(f"Y_train length: {len(Y_train)}")

print(f"Y_test length : {len(Y_test)}")

print(f"Y_val length : {len(Y_val)}")




'''
Resizing the test ,train and validation array

'''

X_train = X_train.reshape(-1, img_size, img_size, 1)
Y_train = np.array(Y_train)

X_val = X_val.reshape(-1, img_size, img_size, 1)
Y_val = np.array(Y_val)

X_test = X_test.reshape(-1, img_size, img_size, 1)
Y_test = np.array(Y_test)

'''
New resized dimensions

'''
print(f"X_train shape new : {X_train.shape}")

print(f"X_test shape new : {X_test.shape}")

print(f"X_val shape new : {X_val.shape}")

print(f"Y_train shape new length: {len(Y_train)}")

print(f"Y_test shape new length: {len(Y_test)}")

print(f"Y_val shape new length: {len(Y_val)}")







print("working...\n")






