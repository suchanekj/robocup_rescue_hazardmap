from config import *

if GPU_NUM > 1:
    import horovod.keras as hvd

import os
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
    if GPU_NUM > 1:
        temp_file = 'model_data/temp{}.h5'.format(hvd.rank())
    else:
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

        if GPU_NUM > 1:
            opt = Adam(lr=lr/10*hvd.size())
            opt = hvd.DistributedOptimizer(opt)
        else:
            opt = Adam(lr=lr/10)
        model.compile(optimizer=opt, loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        if num_train // batch_size // 100 >= 1:
            print("warmup with lr", lr/10)
            model.fit_generator(data_generator_wrapper_sequence(lines[:max(num_train//100, min(10, num_train))], batch_size, input_shape, anchors, num_classes, class_tree),
                                steps_per_epoch=num_train // batch_size // 100,
                                epochs=1,
                                initial_epoch=0,
                                workers=2,
                                max_queue_size=10)

        if GPU_NUM > 1:
            opt = Adam(lr=lr*hvd.size())
            opt = hvd.DistributedOptimizer(opt)
        else:
            opt = Adam(lr=lr)
        model.compile(opt, loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        divider = GPU_NUM
        model.fit_generator(data_generator_wrapper_sequence(lines[:num_train], batch_size, input_shape, anchors, num_classes, class_tree),
                            steps_per_epoch=max(1, num_train // (batch_size * divider)),
                            validation_data=data_generator_wrapper_sequence(lines[num_train:], batch_size, input_shape, anchors,
                                                                   num_classes, class_tree),
                            validation_steps=max(1, num_val // (batch_size * divider)),
                            epochs=current_epoch + epoch - skip,
                            initial_epoch=current_epoch,
                            callbacks=callbacks,
                            workers=2,
                            max_queue_size=10)
        current_epoch += epoch
    return model, current_epoch

def train(specific=None):
    log_dir = 'logs/' + str(TRAINING_CYCLE).zfill(3) + '/'
    classes_path = DATASET_LOCATION + str(0) + '/labelNames.txt'
    anchors_path = 'model_data/yolo_anchors_custom.txt'
    class_names = get_classes(classes_path)
    class_tree = get_class_tree(class_names)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    if GPU_NUM > 1:
        hvd.init()

        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        K.set_session(tf.Session(config=config))

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
            #raise Exception("failed to find latest in logged files")
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

    if GPU_NUM > 1:
        callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0)
        ]
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_delta=TRAINING_PATIENCE_LOSS_MARGIN,
                                      patience=TRAINING_REDUCE_LR_PATIENCE, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=TRAINING_PATIENCE_LOSS_MARGIN,
                                       patience=TRAINING_STOPPING_PATIENCE, verbose=1)

        callbacks = callbacks + [reduce_lr, early_stopping]

        if hvd.rank() == 0:
            checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                         monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
            logging = TensorBoard(log_dir=log_dir, write_images=True)
            callbacks.append(checkpoint)
            callbacks.append(logging)
    else:
        checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                     monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
        logging = TensorBoard(log_dir=log_dir, write_images=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_delta=TRAINING_PATIENCE_LOSS_MARGIN,
                                      patience=TRAINING_REDUCE_LR_PATIENCE, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=TRAINING_PATIENCE_LOSS_MARGIN,
                                       patience=TRAINING_STOPPING_PATIENCE, verbose=1)
        callbacks = [logging, checkpoint, reduce_lr, early_stopping]

    sizes = DATASET_TRAINING_SHAPES
    epochs = TRAINING_EPOCHS
    lrs = TRAINING_LRS

    if TRAIN:
        for i in range(len(sizes)):
            if i <= latest_part:
                current_epoch += np.sum(epochs[i])
                continue
            skip = 0
            for epoch in epochs[i]:
                if skip_epochs >= epoch:
                    skip_epochs -= epoch
                    skip += epoch
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
            if GPU_NUM <= 1:
                model.save_weights(log_dir + 'trained_weights_' + str(i) + '.h5')

    if TEST:
        if TEST_EVALUATE:
            with open(log_dir + "validation.txt", "w") as f:
                f.write("")

        annotation_path = VALIDATION_DATASET_LOCATION + '/labels.txt'
        with open(annotation_path) as f:
            test_lines = f.readlines()
        epochs_to_test_steps = [a for b in TRAINING_EPOCHS for a in b if a != 0]
        epochs_to_test = [sum(epochs_to_test_steps[:i]) for i in range(len(epochs_to_test_steps))]

        for epoch in epochs_to_test:
                files = os.listdir(log_dir)
                if len([f for f in files if ("ep" + str(epoch).zfill(3)) in f]) == 0:
                    print("Missing checkpoint for ep" + str(epoch).zfill(3))
                    continue
                model_file = [f for f in files if ("ep" + str(epoch).zfill(3)) in f][0]

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

                    with open(log_dir + "validation.txt", "a") as f:
                        f.write(f"Epoch: {str(epoch).zfill(3)}\n")
                        # f.write(f"Epoch: {str(epoch).zfill(3)} Validation_score: {result}\n")

                    yolo = YOLO(**settings)

                    def area_box(box):
                        xa, ya, xb, yb = box[0], box[1], box[2], box[3]
                        return (yb-ya)*(xb-xa)

                    def area_intersection(box1, box2):
                        xa, ya, xb, yb = box1[0], box1[1], box1[2], box1[3]
                        xc, yc, xd, yd = box2[0], box2[1], box2[2], box2[3]
                        # why does yolo return top left bottom right?
                        # for training, input was bottom left, top right
                        # xc, yd, xd, yc = box2[0], box2[1], box2[2], box2[3]

                        blx = max(xa, xc)
                        bly = max(ya, yc)
                        trx = min(xb, xd)
                        tr_y = min(yb, yd)
                        if tr_y>bly and trx>blx:
                            return (tr_y-bly)*(trx-blx)
                        return 0

                    def similarity_check(correct_list, predicted_list):
                        ctr = 0
                        for correct_tuple in correct_list:

                            idx = -1
                            best_IOU = 0
                            for i, predicted_tuple in enumerate(predicted_list):
                                correct_label = correct_tuple[-1]
                                correct_box = correct_tuple[:-1]
                                predicted_label = predicted_tuple[-1]
                                predicted_box = predicted_tuple[:-1]
                                predicted_box[1], predicted_box[3] = predicted_box[3], predicted_box[1]
                                if correct_label != predicted_label: continue # skip wrong label

                                area = area_intersection(correct_box, predicted_box)

                                if area == 0: continue # skip wrong bounding box
                                intersection_area = area_intersection(predicted_box, correct_box)
                                intersection_over_union = intersection_area/(area_box(predicted_box)+area_box(correct_box)-intersection_area)
                                print(intersection_area)
                                print(intersection_over_union)
                                print()
                                # get best IOU value
                                if intersection_over_union > best_IOU:
                                    best_IOU = intersection_over_union
                                    idx = i
                            # if found match and IOU>0.5: count that as match, remove match from predictions
                            if idx != -1 and best_IOU > 0.5:
                                del predicted_list[idx]
                                ctr += 1
                        return ctr

                    correct_predicted = 0
                    total_predicted = 0
                    total_relevant = 0

                    for line in test_lines:
                        if line[-1] == '\n':
                            line = line[:-1]
                        lines = line.split(" ")
                        path = lines[0]

                        correct_boxes = [[int(x) for x in txt.split(',')] for txt in lines[1:]]
                        total_relevant += len(correct_boxes)

                        r_image = Image.open(path)

                        predicted_boxes = yolo.detect_boxes(r_image)
                        total_predicted += len(predicted_boxes)

                        ans = similarity_check(correct_boxes, predicted_boxes)
                        correct_predicted += ans

                    with open(log_dir + "validation.txt", "a") as f:

                        f.write(f"Correct Predictions Made By Model: {correct_predicted}\n")
                        f.write(f"Total Predictions Made By Model: {total_predicted}\n")
                        f.write(f"Total Number of Labels: {total_relevant}\n")
                        f.write(f"Precision: {correct_predicted/total_predicted}\n")
                        f.write(f"Recall: {correct_predicted/total_relevant}\n")

                    yolo.close_session()
                    K.clear_session()

                if TEST_VISUALIZE_IMAGES or TEST_VISUALIZE_VIDEO:
                    folder = log_dir + "test" + str(epoch).zfill(3) + "/"
                    if os.path.exists(folder):
                        shutil.rmtree(folder)
                        time.sleep(0.2)
                    os.mkdir(folder)
                    os.mkdir(folder + "imgs/")

                    # settings = {
                    #     "model_path": log_dir + model_file,
                    #     "anchors_path": anchors_path,
                    #     "classes_path": classes_path,
                    #     "score": 0.05,  # 0.3
                    #     "iou": 0.45,  # 0.45
                    # }
                    # K.clear_session()

                    if TEST_VISUALIZE_VIDEO:
                        yolo = YOLO(**settings)
                        detect_video(yolo, "test.mp4", folder[:-1] + ".avi")
                        K.clear_session()

                    if TEST_VISUALIZE_IMAGES:
                        yolo = YOLO(**settings)
                        for line in test_lines[:100]:
                            if line[-1] == '\n':
                                line = line[:-1]
                            line = line.split(" ")[0]
                            r_image = Image.open(line)
                            r_image = yolo.detect_image(r_image)
                            name = line.split("/")[-1]
                            r_image.save(folder + "imgs/" + name)
                        yolo.close_session()
                        K.clear_session()


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
    if GPU_NUM > 1:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        K.set_session(tf.Session(config=config))

    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l],
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
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
    if GPU_NUM > 1:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        K.set_session(tf.Session(config=config))

    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
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
    if n==0 or batch_size<=0: return None
    return DataGenerator(annotation_lines, batch_size, input_shape, anchors, num_classes, class_tree)

if __name__ == '__main__':
    train()
