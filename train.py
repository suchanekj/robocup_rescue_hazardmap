from config import *
from evaluation import evaluate

import os
import copy
import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.layers import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
from keras.utils import Sequence
from PIL import Image
import shutil
import time

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
from yolo import YOLO, detect_video


def train_cycle(model, lrs, epochs, current_epoch, lines, num_train, num_val, input_shape, anchors, num_classes,
                callbacks, class_tree, skip=0):
    yolo_splits = (249, 185, 65, 0)
    temp_file = 'model_data/temp.h5'
    model.save_weights(temp_file)
    model = create_model(input_shape, anchors, num_classes, freeze_body=0, weights_path=temp_file)

    if AVAILABLE_MEMORY_GB == 2:
        batch_size = 1
    else:
        batch_size = 4
        if input_shape[0] * input_shape[1] <= 360 * 480:
            batch_size *= 2
        if input_shape[0] * input_shape[1] <= 240 * 320:
            batch_size *= 2
        if input_shape[0] * input_shape[1] <= 120 * 160:
            batch_size *= 2
    print('Train on {} samples, val on {} samples, with batch size {} for {} epochs.'
          .format(num_train, num_val, batch_size, epochs))
    for lr, epoch, split in zip(lrs, epochs, yolo_splits):
        if skip >= epoch:
            skip -= epoch
            continue
        if split == 65 and 120 * 160 <= input_shape[0] * input_shape[1] <= 240 * 320:
            batch_size = batch_size // 2
        if split == 0:
            batch_size = batch_size // 2
        print("batch_size", batch_size, "lr", lr)

        for i in range(len(model.layers)):
            if isinstance(model.layers[i], BatchNormalization) or i >= split:
                model.layers[i].trainable = True
                continue
            model.layers[i].trainable = False

        opt = Adam(lr=lr / 10)
        model.compile(optimizer=opt, loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        if num_train // batch_size // 100 >= 1:
            print("warmup with lr", lr / 10)
            model.fit_generator(
                data_generator_wrapper_sequence(lines[:max(num_train // 100, min(10, num_train))], batch_size,
                                                input_shape, anchors, num_classes, class_tree),
                steps_per_epoch=num_train // batch_size // 100,
                epochs=1,
                initial_epoch=0,
                workers=2,
                max_queue_size=10)

        opt = Adam(lr=lr)
        model.compile(opt, loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        model.fit_generator(
            data_generator_wrapper_sequence(lines[:num_train], batch_size, input_shape, anchors, num_classes,
                                            class_tree),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator_wrapper_sequence(lines[num_train:], batch_size, input_shape, anchors,
                                                            num_classes, class_tree),
            validation_steps=max(1, num_val // batch_size),
            epochs=current_epoch + epoch - skip,
            initial_epoch=current_epoch,
            callbacks=callbacks,
            workers=2,
            max_queue_size=10)
        current_epoch += epoch
    return model, current_epoch


def test_cycle(log_dir, epochs_to_test, anchors_path, classes_path, test_lines):
    for epoch in epochs_to_test:
        files = os.listdir(log_dir)
        if len([f for f in files if ("ep" + str(epoch).zfill(3)) in f]) == 0:
            print("Missing checkpoint for ep" + str(epoch).zfill(3))
            continue
        model_file = [f for f in files if ("ep" + str(epoch).zfill(3)) in f][0]

        folder = log_dir + "test" + str(epoch).zfill(3) + "/"
        if os.path.exists(folder):
            shutil.rmtree(folder)
            time.sleep(0.2)
        os.mkdir(folder)

        settings = {
            "model_path": log_dir + model_file,
            "anchors_path": anchors_path,
            "classes_path": classes_path,
            "score": 0.05,  # 0.3
            "iou": 0.45,  # 0.45
        }
        K.clear_session()

        if TEST_EVALUATE:
            # model = create_model(input_shape, anchors, num_classes, freeze_body=2,
            #                      weights_path=log_dir + model_file)
            # batch_size = 16
            # model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
            # result = model.evaluate_generator(
            #     data_generator_wrapper_sequence(test_lines, batch_size, input_shape, anchors, num_classes, class_tree),
            #     steps=max(1, len(test_lines) // batch_size))

            for nmx_suppresion in [False, True]:
                with open(log_dir + "validation" + ("_unsurpressed" if not nmx_suppresion else "") + ".txt", "a") as f:
                    f.write(f"Epoch: {str(epoch).zfill(3)}\n")
                    # f.write(f"Epoch: {str(epoch).zfill(3)} Validation_score: {result}\n")

            yolo = YOLO(**settings)

            evaluate(test_lines, yolo, log_dir, epoch)

            yolo.close_session()
            K.clear_session()

        if TEST_VISUALIZE_IMAGES or TEST_VISUALIZE_VIDEO:
            folder = log_dir + "test" + str(epoch).zfill(3) + "/"
            os.mkdir(folder + "imgs/")

            if TEST_VISUALIZE_VIDEO:
                yolo = YOLO(**settings)
                detect_video(yolo, "test.mp4", folder + "video.avi")
                K.clear_session()

            if TEST_VISUALIZE_IMAGES:
                yolo = YOLO(**settings)
                for line in test_lines[:500]:
                    if line[-1] == '\n':
                        line = line[:-1]
                    line = line.split(" ")[0]
                    for nmx_suppresion in [False, True]:
                        r_image = Image.open(line)
                        r_image = yolo.detect_image(r_image, augment=True, nmx_suppresion=nmx_suppresion)
                        name = line.split("/")[-1].split(".")[0] + ("_unsurpressed" if not nmx_suppresion else "") + ".png"
                        r_image.save(folder + "imgs/" + name)
                yolo.close_session()
                K.clear_session()


def test_cycle(log_dir, epochs_to_test, anchors_path, classes_path, test_lines):
    for epoch in epochs_to_test:
        files = os.listdir(log_dir)
        if len([f for f in files if ("ep" + str(epoch).zfill(3)) in f]) == 0:
            print("Missing checkpoint for ep" + str(epoch).zfill(3))
            continue
        model_file = [f for f in files if ("ep" + str(epoch).zfill(3)) in f][0]

        folder = log_dir + "test" + str(epoch).zfill(3) + "/"
        if os.path.exists(folder):
            shutil.rmtree(folder)
            time.sleep(0.2)
        os.mkdir(folder)

        settings = {
            "model_path": log_dir + model_file,
            "anchors_path": anchors_path,
            "classes_path": classes_path,
            "score": 0.05,  # 0.3
            "iou": 0.45,  # 0.45
        }
        K.clear_session()

        if TEST_EVALUATE:
            # model = create_model(input_shape, anchors, num_classes, freeze_body=2,
            #                      weights_path=log_dir + model_file)
            # batch_size = 16
            # model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
            # result = model.evaluate_generator(
            #     data_generator_wrapper_sequence(test_lines, batch_size, input_shape, anchors, num_classes, class_tree),
            #     steps=max(1, len(test_lines) // batch_size))

            for nmx_suppresion in [False, True]:
                with open(log_dir + "validation" + ("_unsurpressed" if not nmx_suppresion else "") + ".txt", "a") as f:
                    f.write(f"Epoch: {str(epoch).zfill(3)}\n")
                    # f.write(f"Epoch: {str(epoch).zfill(3)} Validation_score: {result}\n")
            print(settings)
            yolo = YOLO(**settings)
            print(test_lines)
            evaluate(test_lines, yolo, log_dir, epoch)

            yolo.close_session()
            K.clear_session()

        if TEST_VISUALIZE_IMAGES or TEST_VISUALIZE_VIDEO:
            folder = log_dir + "test" + str(epoch).zfill(3) + "/"
            os.mkdir(folder + "imgs/")

            if TEST_VISUALIZE_VIDEO:
                yolo = YOLO(**settings)
                detect_video(yolo, "test.mp4", folder + "video.avi")
                K.clear_session()

            if TEST_VISUALIZE_IMAGES:
                yolo = YOLO(**settings)
                for line in test_lines[:100]:
                    if line[-1] == '\n':
                        line = line[:-1]
                    line = line.split(" ")[0]
                    for nmx_suppresion in [False, True]:
                        r_image = Image.open(line)
                        r_image = yolo.detect_image(r_image, augment=True, nmx_suppresion=nmx_suppresion)
                        name = line.split("/")[-1].split(".")[0] + ("_unsurpressed" if not nmx_suppresion else "") + ".png"
                        r_image.save(folder + "imgs/" + name)
                yolo.close_session()
                K.clear_session()


def train(specific=None):
    log_dir = 'logs/' + str(TRAINING_CYCLE).zfill(3) + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    classes_path = DATASET_LOCATION + str(0) + '/labelNames.txt'
    anchors_path = 'model_data/yolo_anchors_custom.txt'
    class_names = get_classes(classes_path)
    class_tree = get_class_tree(class_names)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = DATASET_DEFAULT_SHAPE  # multiple of 32, hw

    is_tiny_version = len(anchors) == 6  # default setting
    latest_part = -1
    if specific is None:
        latest = None
        try:
            logged_files = os.listdir(log_dir)
            for part in range(len(DATASET_TRAINING_SHAPES)):
                for file in logged_files:
                    if ("trained_weights_" + str(part)) in file:
                        latest = file
                        latest_part = part
            # for i in range(len(logged_files)):
            #     if logged_files[i][-3:] == ".h5" and logged_files[i][0:2] == "ep":
            #         latest = logged_files[i]
        except:
            # raise Exception("failed to find latest in logged files")
            pass
    else:
        latest = specific

    if latest is None:
        skip_epochs = 0
        current_epoch = 0
        if is_tiny_version:
            model = create_tiny_model(input_shape, anchors, num_classes,
                                      freeze_body=2, weights_path='model_data/yolo-tiny.h5')
            print('Loaded model_data/yolo-tiny.h5')
        else:
            model = create_model(input_shape, anchors, num_classes,
                                 freeze_body=2, weights_path='model_data/yolo_original.h5')
            print('Loaded model_data/yolo_original.h5')
            # model = create_model(input_shape, anchors, num_classes,
            #     freeze_body=2, weights_path='model_data/yolo.h5') # make sure you know what you freeze
    else:
        if specific is None:
            current_epoch = 0
            skip_epochs = 0
        else:
            current_epoch = int(latest[2:5])
            skip_epochs = current_epoch
        model = create_model(input_shape, anchors, num_classes, freeze_body=2,
                             weights_path=log_dir + latest)  # make sure you know what you freeze
        print(log_dir + latest)

    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    logging = TensorBoard(log_dir=log_dir, write_images=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_delta=TRAINING_PATIENCE_LOSS_MARGIN,
                                  patience=TRAINING_REDUCE_LR_PATIENCE, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=TRAINING_PATIENCE_LOSS_MARGIN,
                                   patience=TRAINING_STOPPING_PATIENCE, verbose=1)
    callbacks = [logging, checkpoint, reduce_lr, early_stopping] \
        if not HYPERPARAMETER_SEARCH else [reduce_lr, early_stopping]

    sizes = DATASET_TRAINING_SHAPES
    epochs = TRAINING_EPOCHS
    lrs = TRAINING_LRS

    if TEST:
        if TEST_EVALUATE:
            for nmx_suppresion in [False, True]:
                with open(log_dir + "validation" + ("_unsurpressed" if not nmx_suppresion else "") + ".txt", "w") as f:
                    f.write("")

        validation_annotation_path = VALIDATION_DATASET_LOCATION + '/labels.txt'
        with open(validation_annotation_path) as f:
            test_lines = f.readlines()[:VALIDATION_DATASET_SIZE]

    if TRAIN:
        initial_epoch = 0
        for i in range(len(sizes)):
            if i <= latest_part:
                current_epoch += np.sum(epochs[i])
                initial_epoch += np.sum(epochs[i])
                continue
            skip = 0
            for epoch in epochs[i]:
                if skip_epochs >= epoch:
                    skip_epochs -= epoch
                    skip += epoch
                    initial_epoch += epoch
            if skip >= np.sum(epochs[i]):
                continue
            skip += skip_epochs
            skip_epochs = 0
            annotation_path = DATASET_LOCATION + str(i) + '/labels.txt'
            with open(annotation_path) as f:
                lines = f.readlines()
            np.random.seed(10101)
            np.random.shuffle(lines)
            np.random.seed(None)
            val_split = 1 - DATASET_TRAINING_PART
            num_val = int(len(lines) * val_split)
            num_train = len(lines) - num_val

            model, current_epoch = train_cycle(model, lrs[i], epochs[i], current_epoch, lines, num_train, num_val,
                                               sizes[i], anchors, num_classes,
                                               callbacks, class_tree, skip)

            model.save_weights(log_dir + 'trained_weights_' + str(i) + '.h5')

            if TEST:
                model = None
                time.sleep(0.1)
                epochs_to_test_steps = [a for a in epochs[i] if a != 0]
                epochs_to_test = [initial_epoch + sum(epochs_to_test_steps[:i]) for i in range(len(epochs_to_test_steps))]

                test_cycle(log_dir, epochs_to_test,
                           anchors_path, classes_path, test_lines)
                K.clear_session()

                model = create_model(input_shape, anchors, num_classes, freeze_body=0,
                                     weights_path=log_dir + 'trained_weights_' + str(i) + '.h5')

            initial_epoch = current_epoch

    if TEST and not TRAIN:
        epochs_to_test_steps = [a for b in TRAINING_EPOCHS for a in b if a != 0]
        epochs_to_test = [sum(epochs_to_test_steps[:i+1]) for i in range(len(epochs_to_test_steps))]

        test_cycle(log_dir, epochs_to_test,
                   anchors_path, classes_path, test_lines)



def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    for i in range(len(class_names)):
        class_names[i] = class_names[i].split(" ")[0].zfill(3) + " " + class_names[i].split(" ")[1]
    class_names.sort()
    return class_names


def get_class_tree(_class_names):
    '''loads the class tree'''
    class_names = []
    for c in _class_names:
        class_names.append(c.split(" ")[1])
    superclasses = []
    for i, c in enumerate(class_names):
        superclasses.append([])
        for j, cl in enumerate(class_names):
            if cl in c:
                superclasses[i].append(j)
    return superclasses


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                 weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session()

    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l],
                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num):
                if isinstance(model_body.layers[i], BatchNormalization):
                    continue
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                      weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session()

    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16}[l], w // {0: 32, 1: 16}[l], \
                           num_anchors // 2, num_classes + 5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors // 2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers) - 2)[freeze_body - 1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


class DataGenerator(Sequence):
    def __init__(self, annotation_lines, batch_size, input_shape, anchors, num_classes, class_tree):
        self.annotation_lines = annotation_lines
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.anchors = anchors
        self.num_classes = num_classes
        self.class_tree = class_tree
        self.on_epoch_end()

    def __len__(self):
        return len(self.annotation_lines) // self.batch_size

    def __getitem__(self, idx):
        index = idx % self.__len__()
        image_data = []
        box_data = []
        for b in range(self.batch_size):
            image, box = get_random_data(self.annotation_lines[index + b], self.input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes, self.class_tree)
        return [image_data, *y_true], np.zeros(self.batch_size)

    def on_epoch_end(self):
        np.random.shuffle(self.annotation_lines)


def data_generator_wrapper_sequence(annotation_lines, batch_size, input_shape, anchors, num_classes, class_tree):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    return DataGenerator(annotation_lines, batch_size, input_shape, anchors, num_classes, class_tree)


if __name__ == '__main__':
    train()
