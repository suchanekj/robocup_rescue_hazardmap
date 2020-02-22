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
if TRAIN or TEST:
    train.train()
