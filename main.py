#%%
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import re
import h5py
import pandas as pd

## import the handfeature extractor class
import frameextractor as fe
import handshape_feature_extractor as hfe

# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video
"""
from_directory = os.getcwd() + "\\train_video"
to_directory = os.getcwd() + "\\traindata"
for file in os.listdir(from_directory):
    video_path = from_directory + "\\" + file

    preprocess_gesture_name = file.split("-")[2]
    re_gesture_name = re.compile(r"^([^.]*).*")
    gesture_name = re.match(re_gesture_name, preprocess_gesture_name)
    gesture_name = gesture_name.group(1)

    preprocess_gesture_ref_no = file.split("-")[0]
    re_gesture_ref_no = re.compile(r"^T(.*)")
    gesture_ref_no = re.match(re_gesture_ref_no, preprocess_gesture_ref_no)
    gesture_ref_no = int(gesture_ref_no.group(1)) - 2
    png_path = to_directory + "\\" + gesture_name + "\\"
    fe.frameExtractor(video_path, png_path, gesture_ref_no)
"""
imagepaths_train = []

for root, dirs, files in os.walk(".", topdown=False): 
  if "traindata" in root:
     for name in files:
       path = os.path.join(root, name)
       print(path)
       if path.endswith("png"): # We want only the images
         imagepaths_train.append(path)

print("imagepaths_train:", imagepaths_train)

train_fv = []

extractor = hfe.HandShapeFeatureExtractor.get_instance()
for path in imagepaths_train:
   fv = extractor.extract_feature(path)
   train_fv.append(fv)
   print(fv.shape)

train_fv = np.array(train_fv)
#Because FanOn and FanOff are swiched in the result.csv
train_fv[[33, 36]] = train_fv[[36, 33]]
train_fv[[34, 37]] = train_fv[[37, 34]]
train_fv[[35, 38]] = train_fv[[38, 35]]
print(train_fv)

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video
"""
from_directory = os.getcwd() + "\\test_video"
to_directory = os.getcwd() + "\\test"
for file in os.listdir(from_directory):
    count = re.findall(r"\d+", file)[0]
    video_path = from_directory + "\\" + file

    preprocess_gesture_name = file.split("-")[2]
    re_gesture_name = re.compile(r"^([^.]*).*")
    gesture_name = re.match(re_gesture_name, preprocess_gesture_name)
    gesture_name = gesture_name.group(1)
    
    preprocess_gesture_ref_no = file.split("-")[0]
    re_gesture_ref_no = re.compile(r"^T(.*)")
    gesture_ref_no = re.match(re_gesture_ref_no, preprocess_gesture_ref_no)
    gesture_ref_no = int(gesture_ref_no.group(1)) - 2
    png_path = to_directory + "\\" + gesture_name + "\\"
    fe.frameExtractor(video_path, png_path, gesture_ref_no)
"""

imagepaths_test = []

for root, dirs, files in os.walk(".", topdown=False): 
  if "test\\" in root:
     for name in files:
       path = os.path.join(root, name)
       print(path)
       if path.endswith("png"): # We want only the images
         imagepaths_test.append(path)

print("imagepath_test:", imagepaths_test)


test_fv = []

extractor = hfe.HandShapeFeatureExtractor.get_instance()
for path in imagepaths_test:
   fv = extractor.extract_feature(path)
   test_fv.append(fv)
   print(fv)

test_fv = np.array(test_fv)
#Because FanOn and FanOff are swiched in the result.csv
test_fv[[33, 36]] = test_fv[[36, 33]]
test_fv[[34, 37]] = test_fv[[37, 34]]
test_fv[[35, 38]] = test_fv[[38, 35]]
print(test_fv)

# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

df_result = pd.DataFrame(columns=["GestureName","T1_Label","T1_CS","T2_Label","T2_CS","T3_Label","T3_CS"])
df_result["GestureName"] = pd.Series(["0","1","2","3","4","5","6","7","8","9","DecreaseFanSpeed","FanOn","FanOff","IncreaseFanSpeed","LightOff","LightOn","SetThermo"])
df_result = df_result.set_index("GestureName")
print(df_result)

m = tf.keras.metrics.CosineSimilarity(axis=1)
df_result_arr = []
for i in test_fv:
  temp_arr = []
  print(temp_arr)
  for j in train_fv:
     m.update_state(i, j)
     temp_arr.append(m.result().numpy())
  print(temp_arr)
  temp_arr = np.array(temp_arr)
  argmin_cs = np.argmin(temp_arr)//3
  df_result_arr.append(argmin_cs)
  min_cs = np.min(temp_arr)
  df_result_arr.append(min_cs)

print(df_result_arr)
for k in range(len(df_result_arr)):
  df_result.iloc[k//6, k%6] = df_result_arr[k]  

print(df_result)  
"""
hf = h5py.File('cnn_model.h5', 'r')
print(list(hf.keys()))
model_weights = hf['model_weights']
optimizer_weights = hf['optimizer_weights']
for key, value in hf['optimizer_weights']['training']['Adam'].items():
    print(key, ' : ', value)
"""


