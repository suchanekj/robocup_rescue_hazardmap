import keras.backend as K

from evaluation import *
from yolo import YOLO

test_lines = []
for i in range(5):
    path = f'datasets/dataset_mixed_{i}/labels.txt'
    with open(path) as f:
        test_lines += f.readlines()
K.clear_session()
settings = {
    'model_path': 'logs/021/ep085-loss37.146-val_loss37.585.h5',
    'anchors_path': 'model_data/yolo_anchors_custom.txt',
    'classes_path': 'datasets/dataset_0/labelNames.txt',
    'score': 0.05,
    'iou': 0.45
}
yolo = YOLO(**settings)
test_lines = test_lines
evaluate(test_lines, yolo, 'test_results/', 0)
