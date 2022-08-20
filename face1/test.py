import pickle
def _save_pickle(obj, file_path):
  with open(file_path, 'wb') as f:
    pickle.dump(obj, f)

def _load_pickle(file_path):
  with open(file_path, 'rb') as f:
    obj = pickle.load(f)
  return obj

y_label=_load_pickle("Dataset/y_labels.pkl")
image_file=_load_pickle("Dataset/images_file.pkl")

print(image_file)