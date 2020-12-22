import urllib
import os

if __name__=="__main__":
    if not os.path.isfile(MODEL_LOCATION):
        import convert
        urllib.request.urlretrieve("https://pjreddie.com/media/files/yolov3.weights", "model_data/yolov3.weights")
        convert.convert("yolov3.cfg", "model_data/yolov3.weights", MODEL_LOCATION)

    if MAKE_DATASET:
        from robocup_rescue_hazardmap import synthetic_dataset

        synthetic_dataset.createDataset()
    if TRAIN or TEST:
        import train
        train.train()
