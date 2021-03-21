from PIL import Image
import keras.backend as K
from yolo import YOLO
"""
Requirements:
python 3.7.6
tensorflow 1.14
keras 2.2.4
"""
K.clear_session()
settings = {
    'model_path': 'model_data/ep085-loss37.146-val_loss37.585.h5',
    'anchors_path': 'model_data/yolo_anchors_custom.txt',
    'classes_path': 'model_data/labelNames.txt',
    'score': 0.05,
    'iou': 0.45
}
# validation_annotation_path = VALIDATION_DATASET_LOCATION + '/labels.txt'
# with open(validation_annotation_path) as f:
#     test_lines = f.readlines()[:VALIDATION_DATASET_SIZE]

test_images = ['37.png', '39.png']
yolo = YOLO(**settings)
for path in test_images:
    r_image = Image.open(path)
    boxes = yolo.detect_boxes(r_image, True, nmx_suppresion=True)
    img = yolo.detect_image(r_image, True, nmx_suppresion=True)
    print(boxes)
