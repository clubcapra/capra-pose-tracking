from typing import Collection
import numpy as np
from ..constants import KEYPOINT_OF_INTEREST

def convertToOneHot(y, nb_outputs, pose_dict):
  oh_y = np.zeros((y.shape[0], nb_outputs))

  for i in range(y.shape[0]):
    index = pose_dict[y[i]]
    oh_y[i][index] = 1

  return oh_y

def convertLabelsToInt(y, pose_dict):
  y_int = np.zeros_like(y, dtype=int)

  for i in range(y.shape[0]):
    y_int[i] = pose_dict[y[i]]

  return y_int

def convertSoftmaxToIndex(y):
  return np.argmax(y, axis=1)

# def balanceDataset(X, y):
#   unique, counts = np.unique(y, return_counts=True)
#   ledger = dict(zip(unique, counts))
#   print(ledger)
#   minLabel = unique[np.argmin(counts)]
#   minCount = np.min(counts)
  
#   it = np.nditer(y, flags=['f_index'])
#   for label in it:
#     print(label)
#     print(ledger.get(str(label)))
#     print(minCount)
#     if label is not minLabel and ledger.get(str(label)) > minCount:
#       X = np.delete(X, it.index)
#       y = np.delete(y, it.index)
#       ledger[str(label)] = ledger[str(label)] - 1

#   return X, y 

def balanceDataset(X, y):
  unique, counts = np.unique(y, return_counts=True)
  minLabel = unique[np.argmin(counts)]
  minCount = np.min(counts)
  unique = np.delete(unique, np.where(unique == minLabel))
  for label in unique:
    indexes = np.flatnonzero(y == label)
    indexesToRemove = indexes[minCount:]
    print("x shape before balancing:", X.shape)
    X = np.delete(X, indexesToRemove, axis=0)
    y = np.delete(y, indexesToRemove)

  return X, y

def normalize_skeleton(keypoints):

  # Calculate the center of the body (average of all keypoints)
  body_center = np.mean(keypoints, axis=0)

  # Center all keypoints by subtracting the body center
  normalized_keypoints = keypoints - body_center

  return normalized_keypoints

def getKeypointsOfInterestFromBodyData(bodyData: Collection[float]):
    data = []
    keypoints = normalize_skeleton([bodyData[x] for x in KEYPOINT_OF_INTEREST])
    data.append(keypoints)

    return np.array(data)
    
  


