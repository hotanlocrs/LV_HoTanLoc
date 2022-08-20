import cv2
import os
import numpy as np
import pickle

def _save_pickle(obj, file_path):
  with open(file_path, 'wb') as f:
    pickle.dump(obj, f)

def _load_pickle(file_path):
  with open(file_path, 'rb') as f:
    obj = pickle.load(f)
  return obj
y_label=_load_pickle("Dataset/y_labels.pkl")
def read_name(y_label):
  name=[]
  for t in y_label:
    s=''
    for i in range(8,len(t)):
      if ord(t[i+1])>60:
        s=s+t[i]
      if ord(t[i+1])<=60:
        break
    name.append(s)
  return name
name_face=read_name(y_label)
print(len(name_face))
print(name_face[10])
_save_pickle(name_face,"Dataset/name_face.pkl")
