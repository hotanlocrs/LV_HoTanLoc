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

from sklearn.metrics.pairwise import cosine_similarity


def _most_similarity(embed_vecs, vec, labels):
  sim = cosine_similarity(embed_vecs, vec)
  sim = np.squeeze(sim, axis = 1)
  argmax = np.argsort(sim)[::-1][:1]
  label = [labels[idx] for idx in argmax][0]
  return label

X_test= _load_pickle("Dataset/X_test.pkl")
X_train= _load_pickle("Dataset/X_train.pkl")
y_train= _load_pickle("Dataset/y_train.pkl")
y_test= _load_pickle("Dataset/y_test.pkl")

print(X_test[1])

vec = X_test[1].reshape(1, -1)
vec_train=X_train.reshape(-1,1)

print(_most_similarity(vec_train, vec, y_train))

from sklearn.metrics import accuracy_score

y_preds = []
for vec in X_test:
  vec = vec.reshape(1, -1)
  y_pred = _most_similarity(vec_train, vec, y_train)
  y_preds.append(y_pred)

print(accuracy_score(y_preds, y_test))





