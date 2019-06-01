import urllib
import os

import dataset
import train
from config import *
import convert

if not os.path.isfile(MODEL_LOCATION):
    urllib.request.urlretrieve("https://pjreddie.com/media/files/yolov3.weights", "model_data/yolov3.weights")
    convert.convert("yolov3.cfg", "model_data/yolov3.weights", MODEL_LOCATION)


if MAKE_DATASET:
    dataset.createDataset()
if TRAIN:
    train.train()

"""

  605/80001 [..............................] - ETA: 8:47:47 - loss: 272.4898D:\suchanek\programming\robocup_rescue_hazardmap\datasets\dataset\0090776.png 50,75,426,359,12


  606/80001 [..............................] - ETA: 8:47:46 - loss: inf     D:\suchanek\programming\robocup_rescue_hazardmap\datasets\dataset\0095423.png 430,36,583,233,19


  607/80001 [..............................] - ETA: 8:47:45 - loss: nanD:\suchanek\programming\robocup_rescue_hazardmap\datasets\dataset\0096311.png 334,0,599,217,18

"""