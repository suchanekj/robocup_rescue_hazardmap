import urllib
import os

from config import *


if __name__ == "__main__":
    if not os.path.isfile(MODEL_LOCATION):
        import convert
        urllib.request.urlretrieve("https://pjreddie.com/media/files/yolov3.weights", "model_data/yolov3.weights")
        convert.convert("yolov3.cfg", "model_data/yolov3.weights", MODEL_LOCATION)

    if MAKE_DATASET:
        import manual_dataset
        manual_dataset.createDataset()
        import synthetic_dataset
        synthetic_dataset.createDataset(extend=True)
    if TRAIN or TEST:
        import train
        train.train("ep021-loss42.117-val_loss40.874.h5")
