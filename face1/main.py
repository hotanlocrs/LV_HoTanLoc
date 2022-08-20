import cv2
import os
import numpy as np
import pickle
path=''
EMBEDDING_FL = os.path.join(path,"openface.nn4.small2.v1.t7")
def _load_torch(model_path_fl):
  model = cv2.dnn.readNetFromTorch(model_path_fl)
  return model
encoder = _load_torch(EMBEDDING_FL)

def _blobImage(image, out_size = (300, 300), scaleFactor = 1.0, mean = (104.0, 177.0, 123.0)):
  imageBlob = cv2.dnn.blobFromImage(image,
                                    scalefactor=scaleFactor,   # Scale image
                                    size=out_size,  # Output shape
                                    mean=mean,  # Trung bình kênh theo RGB
                                    swapRB=False,  # Trường hợp ảnh là BGR thì set bằng True để chuyển qua RGB
                                    crop=False)
  return imageBlob


from face_recognition import face_locations

def _image_read(image_path):
  image = cv2.imread(image_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image


def _extract_bbox(image, single = True):
  bboxs = face_locations(image)
  if len(bboxs)==0:
    return None
  if single:
    bbox = bboxs[0]
    return bbox
  else:
    return bboxs


def _extract_face(image, bbox, face_scale_thres = (20, 20)):

  h, w = image.shape[:2]
  try:
    (startY, startX, endY, endX) = bbox
  except:
    return None
  minX, maxX = min(startX, endX), max(startX, endX)
  minY, maxY = min(startY, endY), max(startY, endY)
  face = image[minY:maxY, minX:maxX].copy()

  (fH, fW) = face.shape[:2]

  if fW < face_scale_thres[0] or fH < face_scale_thres[1]:
    return None
  else:
    return face

def _model_processing(image):
  faces = []
  (h, w) = image.shape[:2]
  # Detect vị trí các khuôn mặt trên ảnh. Gỉa định rằng mỗi bức ảnh chỉ có duy nhất 1 khuôn mặt của chủ nhân classes.
  bbox =_extract_bbox(image, single=True)
  # print(bbox)
  if bbox is not None:
    # Lấy ra face
    face = _extract_face(image, bbox, face_scale_thres = (20, 20))
    if face is not None:
      faces.append(face)
  return faces

def _load_pickle(file_path):
  with open(file_path, 'rb') as f:
    obj = pickle.load(f)
  return obj


def _embedding_faces(encoder, faces):
  emb_vecs = []
  for face in faces:
    faceBlob = _blobImage(face, out_size = (96, 96), scaleFactor=1/255.0, mean=(0, 0, 0))
    # Embedding face
    encoder.setInput(faceBlob)
    vec = encoder.forward()
    emb_vecs.append(vec)
  return emb_vecs


from sklearn.metrics.pairwise import cosine_similarity
def _most_similarity(embed_vecs, vec):
  sim = cosine_similarity(embed_vecs, vec)
  sim = np.squeeze(sim, axis = 1)
  argmax = np.argsort(sim)[::-1][:1]
  return argmax

def test_image(image):
  faces = _model_processing(image)
  embed=_embedding_faces(encoder,faces)
  em_loc = _load_pickle("Dataset/embed_blob_faces.pkl")
  vec = embed[0]
  vec_train = []
  for t in em_loc:
    vec_train.append(t[0])
  k=_most_similarity(vec_train, vec)
  name_face=_load_pickle("Dataset/name_face.pkl")
  return name_face[k[0]]

image= _image_read("sontung.JPG")
print(test_image(image))
