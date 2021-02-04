from PIL import Image
import keras.backend as K
from yolo import YOLO

log_dir = 'logs/016/'
model_file = 'ep085-loss21.545-val_loss21.643.h5'
anchors_path = 'model_data/yolo_anchors_custom.txt'
classes_path = 'datasets2/dataset_open_0/labelNames.txt'

settings = {
    "model_path": log_dir + model_file,
    "anchors_path": anchors_path,
    "classes_path": classes_path,
    "score": 0.05,  # 0.3
    "iou": 0.45,  # 0.45
}

line = '/datasets2/dataset_open_4/0.png'
destination = '/test_results/test/png'

K.clear_session()
yolo = YOLO(**settings)
r_image = Image.open(line)
r_image = yolo.detect_image(r_image)
r_image.save(destination)
yolo.close_session()
K.clear_session()
