from google_images_download import google_images_download
import os
import shutil
from PIL import Image
import numpy as np
import cv2
import copy
import string
import time
import random
import datetime
import sys
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import pyqrcodeng as pyqrcode
import pandas as pd
from skimage.measure import regionprops
from collections.abc import Iterable
import threading
import subprocess
import multiprocessing as mp
from queue import Empty as qEmpty
import psutil

from config import *


objectList = []
objectImgs = []
objectNames = []
objectIDs = {}
baseList = []
baseFiles = []
baseDefaultNamesPositions = []
doorBases = []
personBases = []
nothingBases = []
objectNums = {}
objectTree = None

# Door - /m/02dgv
# Fire extinguisher - folder
# Baby doll - folder
# Person - /m/03bt1vf /m/04yx4 /m/01g317
# Valve - folder
# Hazmat labels - images
# Qr codes - script -> folder
# Exit signs - folder
# Fire extinguisher sign - folder


def rand(a=0., b=1.):
    return np.random.rand()*(b-a) + a


def newDownload():
    if not os.path.exists("openimages/validation"):
        subprocess.run(["aws", "s3", "--no-sign-request", "sync",
                        "s3://open-images-dataset/validation", "openimages/validation"])
    if not os.path.exists("openimages/test"):
        subprocess.run(["aws", "s3", "--no-sign-request", "sync",
                        "s3://open-images-dataset/test", "openimages/test"])


def generate_qr_codes():
    output_folder = "objects/qr_code/"
    shutil.rmtree(output_folder, ignore_errors=True)
    time.sleep(0.1)
    os.makedirs(output_folder)
    for i in range(DATASET_NUM_IMAGES // 100):
        while True:
            try:
                size = random.randint(1, 2) * random.randint(1, 2) * random.randint(1, 2) * random.randint(1, 5)
                text = ''.join(random.choice(string.ascii_letters)
                               for i in range(random.randint(1, int(size ** 2.0 * 15))))
                qrcode = pyqrcode.create(text, version=size)
                break
            except Exception as e:
                print(e)
        qrcode.png(output_folder + str(i) + ".png", scale = 200 // size)


def filterObjects(size_step):
    print("Filtering")
    object_fs = []
    for root, dirs, files in os.walk("objects", topdown=False):
        for name in files:
            file = os.path.join(root, name).replace('\\', '/')
            object_fs.append(file)
    shutil.rmtree("filtered_objects/", ignore_errors=True)
    time.sleep(0.5)
    os.makedirs("filtered_objects/")
    for obj in object_fs:
        filtering = (0, 0)
        for key in DATASET_OBJECT_BACKGROUND_REMOVAL.keys():
            if key in obj:
                filtering = DATASET_OBJECT_BACKGROUND_REMOVAL[key]
        f = obj
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None or not hasattr(img, "shape"):
            print("Failed on", f)
            continue
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
        h, w, ch = img.shape
        if ch == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
            img[:, :, 3] = 255
        if "hazmat" in obj and "hazmat_other" not in obj:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), 45, 1)
            img = cv2.warpAffine(img, M, (w, h))
            img[:,:,0] *= np.clip(img[:,:,3], 0, 1)
            img[:,:,1] *= np.clip(img[:,:,3], 0, 1)
            img[:,:,2] *= np.clip(img[:,:,3], 0, 1)

        if filtering[0]:
            if np.average(img[0, 0, :3]) > 230:
                img_rgb = img[:, :, :3].copy()
                cv2.floodFill(img_rgb, np.zeros((h + 2, w + 2), np.uint8), (0, 0), (255, 255, 255),
                              loDiff=(DATASET_OBJECT_BACKGROUND_STEP,)*3, upDiff=(DATASET_OBJECT_BACKGROUND_STEP,)*3,
                              flags=8)
                img[:, :, :3] = img_rgb
            background = np.logical_or(np.logical_and(np.logical_and(img[:, :, 0] >= 254, img[:, :, 1] >= 254),
                                                      img[:, :, 2] >= 254), img[:, :, 3] <= 250)
            background = 255 - 255 * background
            mask = np.zeros((h + 2, w + 2), np.uint8)
            mask[1:-1, 1:-1] = background

            background = np.zeros((h, w), np.uint8)
            cv2.floodFill(background, mask, (0, 0), 1)
            img[:, :, 3] = (1 - background) * img[:, :, 3]
        if filtering[1]:
            background = np.logical_or(np.logical_and(np.logical_and(img[:, :, 0] >= 254, img[:, :, 1] >= 254),
                                                      img[:, :, 2] >= 254), img[:, :, 3] <= 250)
            img[:, :, 3] = 255 - 255 * np.asarray(background, np.uint8)

        x0, x1 = h + w, 0
        y0, y1 = h + w, 0
        props = regionprops(img)
        for p in props:
            # print(p.bbox)
            x0 = min(p.bbox[0], x0)
            x1 = max(p.bbox[3], x1)
            y0 = min(p.bbox[1], y0)
            y1 = max(p.bbox[4], y1)

        if "hazmat" in f:
            a = max(x1 - x0, y1 - y0)
            res = np.ones((a, a, 4), np.uint8) * 255
            res[:, :, 3] = 0

            res[(a - x1 + x0) // 2: (a + x1 - x0) // 2, (a - y1 + y0) // 2: (a + y1 - y0) // 2] = img[x0:x1, y0:y1]

            res = cv2.resize(res, (DATASET_TRAINING_OBJECT_SIZES[size_step], DATASET_TRAINING_OBJECT_SIZES[size_step]))

        else:
            res = img[x0:x1, y0:y1]

            scale_h = DATASET_TRAINING_OBJECT_SIZES[size_step] / res.shape[0]
            scale_w = DATASET_TRAINING_OBJECT_SIZES[size_step] / res.shape[0]
            scale = min(scale_h, scale_w)
            new_shape = tuple(map(int, np.round([res.shape[1] * scale, res.shape[0] * scale])))

            res = cv2.resize(res, new_shape)

        dir = "filtered_" + "/".join(f.split("/")[:-1])
        if not os.path.exists(dir):
            os.makedirs(dir)
        cv2.imwrite("filtered_" + ".".join(f.split(".")[:-1]) + ".png", res)


def filterOpenImages():
    filtered_df = pd.DataFrame()
    banned_df = pd.DataFrame()
    parts = ["validation", "test"]

    for part in parts:
        annotation_df = pd.read_csv("openimages/" + part + "-annotations-bbox.csv")
        for i in annotation_df.index[:]:
            if annotation_df["LabelName"][i] in DATASET_OPENIMAGES_FORBIDDEN_LABELS or \
                    annotation_df["LabelName"][i] in DATASET_OPENIMAGES_LABEL_TO_OBJECT.keys() and \
                    (int(annotation_df["IsGroupOf"][i]) != 0 or int(annotation_df["IsDepiction"][i]) != 0):
                banned_df = pd.concat([banned_df, pd.DataFrame(data=annotation_df.loc[i]).T])

            if annotation_df["LabelName"][i] in DATASET_OPENIMAGES_LABEL_TO_OBJECT.keys() and \
                    int(annotation_df["IsGroupOf"][i]) == 0 and int(annotation_df["IsDepiction"][i]) == 0:
                filtered_df = pd.concat([filtered_df, pd.DataFrame(data=annotation_df.loc[i]).T])
            if i % 1000 == 0:
                print(i, len(filtered_df))

    print(np.unique(filtered_df["LabelName"], return_counts=True))
    filtered_df.to_csv("openimages/all-annotations-bbox.csv")
    num = 0
    for part in parts:
        files = os.listdir("openimages/" + part)
        for file in files:
            if file[:-4] not in filtered_df["ImageID"].values and \
                    (part != "validation" or file[:-4] in banned_df["ImageID"].values):
                os.remove("openimages/" + part + "/" + file)
            else:
                num += 1
    print(num)


class ClassTree(object):
    def __init__(self, pth):
        while len(pth) > 0 and pth[-1] == "/":
            pth = pth[:-1]
        self.pth = pth
        self.pth_l = pth.split("/")
        if len(pth) == 0:
            self.pth_l = []
        self.children = []

    def insert(self, pth):
        while pth[-1] == "/":
            pth = pth[:-1]
        if self.pth not in pth:
            return False
        if self.pth == pth:
            return True
        for c in self.children:
            if c.insert(pth):
                return True
        one_further = "/".join(pth.split("/")[:len(self.pth_l)+1])
        self.children.append(ClassTree(one_further))
        self.children[-1].insert(pth)
        return True

    def get(self):
        if len(self.children) == 0:
            return self.pth
        else:
            return random.choice(self.children).get()


def makeObjectList(size_step):
    global objectImgs
    global objectList
    global objectNames
    global objectIDs
    global objectTree
    object_fs = []
    super_object_fs = []
    objectTree = ClassTree("")
    for root, dirs, files in os.walk("filtered_objects", topdown=False):
        for name in files:
            file = "/".join(os.path.join(root, name).replace('\\', '/').split("/")[1:])
            if "!" not in file:
                object_fs.append(file)
        for name in dirs:
            file = "/".join(os.path.join(root, name).replace('\\', '/').split("/")[1:])
            if "!" in file:
                object_fs.append(file)
            else:
                super_object_fs.append(file)
    dataset_f = DATASET_LOCATION + str(size_step)
    names_f = open(dataset_f + "/labelNames.txt", "w")
    for obj in list(set(DATASET_OPENIMAGES_LABEL_TO_OBJECT.values())):
        objectTree.insert(obj)
    for obj in object_fs:
        objectImgs.append([])
        name = obj.split(".")[0]
        objectTree.insert(name)
        objectNames.append(name)
        objectIDs[name] = len(objectNames) - 1
        names_f.write(str(len(objectNames) - 1) + " " + name + "\n")
        if os.path.isdir("filtered_objects/" + obj):
            obj_fs = ["filtered_objects/" + obj + "/" + f for f in os.listdir("filtered_objects/" + obj)]
        else:
            obj_fs = ["filtered_objects/" + obj]
        for obj_f in obj_fs:
            img = cv2.imread(obj_f, cv2.IMREAD_UNCHANGED)
            objectImgs[-1].append(img)
    num_id = len(objectNames) + len(list(set(DATASET_OPENIMAGES_LABEL_TO_OBJECT.values())))
    for obj in super_object_fs:
        name = obj.split(".")[0]
        names_f.write(str(num_id) + " " + name + "\n")
        num_id += 1
    hazmat_skip = 0
    for i in range(int(DATASET_MAX_OBJECTS_PER_IMG * DATASET_NUM_IMAGES * 4) + 100):
        obj = None
        while objectIDs.get(obj) is None:
            obj = objectTree.get()
        if "hazmat" in obj and "hazmat_other" not in obj:
            if hazmat_skip > 0:
                hazmat_skip -= 1
                continue
            if rand() < DATASET_FOURS_PART:
                hazmat_skip = 3
                obj = [objectIDs[obj]]
                while len(obj) != 4:
                    new_obj = objectTree.get()
                    if objectIDs.get(new_obj) is not None and "hazmat" in new_obj:
                        obj.append(objectIDs[new_obj])
            else:
                obj = objectIDs[obj]
        else:
            obj = objectIDs[obj]
        objectList.append(obj)
    names_f.close()


def makeBaseList(size_step):
    global baseList
    global baseFiles
    global baseDefaultNamesPositions
    global doorBases, personBases, nothingBases
    dataset_f = DATASET_LOCATION + str(size_step)
    names_f = open(dataset_f + "/labelNames.txt", "a")

    parts = ["validation", "test"]
    all_df = pd.read_csv("openimages/all-annotations-bbox.csv", index_col=0)
    doors = 0
    persons = 0

    objectFromBaseNames = list(set(DATASET_OPENIMAGES_LABEL_TO_OBJECT.values()))
    objectNames.extend(objectFromBaseNames)
    for objName in objectFromBaseNames:
        objectIDs[objName] = len(objectIDs.keys())
        names_f.write(str(objectIDs[objName]) + " " + objName + "\n")

    id = 0
    if len(baseDefaultNamesPositions) == 0:
        for part in parts:
            files = os.listdir("openimages/" + part)
            for f in files:
                if DEBUG and min(len(doorBases), len(nothingBases), len(personBases)) >= 1:
                    continue
                if min(len(doorBases), len(nothingBases), len(personBases)) >= \
                        DATASET_NUM_IMAGES // (10 * DATASET_CREATION_THREADS):
                    continue
                name = f[:-4]
                objects_df = all_df[all_df["ImageID"] == name]
                if len(objects_df) > 3:
                    continue
                print(".", end="")
                sys.stdout.flush()
                if (id + 1) % 100 == 0:
                    print(id)
                    if random.random() < 0.2:
                        print("cpu", psutil.cpu_percent(interval=None, percpu=True))
                if len(objects_df) == 0:
                    nothingBases.append(id)
                elif DATASET_OPENIMAGES_LABEL_TO_OBJECT[objects_df["LabelName"].values[0]] == "physical/door":
                    doorBases.append(id)
                else:
                    personBases.append(id)

                rows, cols, ch = cv2.imread("openimages/" + part + "/" + f, cv2.IMREAD_COLOR).shape
                namesPositions = []
                for i in objects_df.index:
                    object_name = DATASET_OPENIMAGES_LABEL_TO_OBJECT[objects_df["LabelName"][i]]
                    x0 = int(cols * objects_df["XMin"][i])
                    x1 = int(cols * objects_df["XMax"][i])
                    y0 = int(rows * objects_df["YMin"][i])
                    y1 = int(rows * objects_df["YMax"][i])
                    namesPositions.append([object_name, x0, y0, x1, y1])

                baseFiles.append("openimages/" + part + "/" + f)
                baseDefaultNamesPositions.append(namesPositions)

                id += 1

    print("loaded, making list")
    object_num = 0
    door_skip = 0
    person_skip = 0
    objects_per_img = np.sum(np.multiply(DATASET_OBJECT_PLACE_CHANCE, range(DATASET_MAX_OBJECTS_PER_IMG + 1)))
    for i in range(DATASET_NUM_IMAGES * 2):
        print(".", end="")
        if (i + 1) % 100 == 0:
            print(i)
        object_num += objects_per_img
        chosen_base = None
        while object_num >= 1:
            object_num -= 1
            obj = objectTree.get()
            if "person" in obj:
                person_skip -= 1
                if person_skip > 0:
                    continue
                chosen_base = random.choice(doorBases)
                break
            if "door" in obj:
                door_skip -= 1
                if door_skip > 0:
                    continue
                chosen_base = random.choice(personBases)
                break
        if chosen_base is None:
            chosen_base = random.choice(nothingBases)

        for namePosition in baseDefaultNamesPositions[chosen_base]:
            if namePosition[0] == "physical/door":
                doors += 1
                door_skip += 1
            if namePosition[0] == "physical/person":
                persons += 1
                person_skip += 1
        baseList.append(chosen_base)
    print("doors:", doors, "people:", persons)
    names_f.close()


def crop(x, crop_size):
    h, w = x.shape[:2]
    x0 = int(h * crop_size / 2)
    x1 = h - x0
    y0 = int(w * crop_size / 2)
    y1 = w - y0
    return x[x0:x1, y0:y1]


def name_to_config_key(name):
    for key in DATASET_OBJECT_CROP_STRENGTH.keys():
        if key in name:
            return key
    print(name, DATASET_OBJECT_CROP_STRENGTH.keys(), "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


def objectMake(obj, size_step):
    strength_mod = DATASET_FILTERING_STRENGTHS[size_step]
    out = [None, None]
    if isinstance(obj, Iterable):
        crop_strength = strength_mod * DATASET_OBJECT_CROP_STRENGTH["hazmat"]
        x = []
        crop_1 = rand(0, crop_strength) if rand() < 0.34 * (0.5 + crop_strength) else 0
        crop_2 = rand(0, crop_strength) if rand() < 0.34 * (0.5 + crop_strength) else 0
        crop_3 = rand(0, crop_strength) if rand() < 0.34 * (0.5 + crop_strength) else 0
        for j in range(4):
            a = copy.deepcopy(random.choice(objectImgs[obj[j]]))
            a = crop(a, crop_1)
            rows, cols, colors = a.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2),
                                        90 * int(rand() * 4) - DATASET_FOUR_MEMBER_WIGGLE
                                        + int(rand() * (1 + DATASET_FOUR_MEMBER_WIGGLE * 2)), 1)
            a = cv2.warpAffine(a, M, (cols, rows))
            M = np.float32([[1, 0, - DATASET_FOUR_MEMBER_WIGGLE * 2 +
                             int(rand() * (4 * DATASET_FOUR_MEMBER_WIGGLE + 1))],
                            [0, 1, - DATASET_FOUR_MEMBER_WIGGLE * 2 +
                             int(rand() * (4 * DATASET_FOUR_MEMBER_WIGGLE + 1))]])
            a = cv2.warpAffine(a, M, (cols, rows))
            a = crop(a, crop_2)
            x.append(a)

        cat0 = np.concatenate((x[0], x[1]), axis=0)
        cat1 = np.concatenate((x[2], x[3]), axis=0)
        four = np.concatenate((cat0, cat1), axis=1)
        four = crop(four, crop_3)

        name = objectNames[obj[0]] + ";" + objectNames[obj[1]] + ";" + \
               objectNames[obj[2]] + ";" + objectNames[obj[3]]

        out[0] = four
        out[1] = name
    else:
        out[1] = objectNames[obj]
        crop_strength = strength_mod * DATASET_OBJECT_CROP_STRENGTH[name_to_config_key(out[1])]
        crop_x = rand(0, crop_strength) if rand() < (0.5 + crop_strength) else 0
        img = random.choice(objectImgs[obj])
        img = copy.deepcopy(img)
        out[0] = crop(img, crop_x)

    return out


def objectFilter(srcA, name, size_step):
    strength_mod = DATASET_FILTERING_STRENGTHS[size_step]
    rows, cols, colors = srcA.shape
    srcA = np.asarray(srcA, np.uint8)

    config_key = name_to_config_key(name)

    if rand() < 0.9 * strength_mod * DATASET_OBJECT_COLOUR_FILTER_STRENGTH[config_key]:
        strength = strength_mod * DATASET_OBJECT_COLOUR_FILTER_STRENGTH[config_key]
        noise = np.zeros((rows, cols, 4))
        m = np.array([255 * rand(), 255 * rand(), 255 * rand(), 40 * strength * rand()])
        sigma = np.array([80 * rand(), 80 * rand(), 80 * rand(), 4 * strength * rand()])
        cv2.randn(noise, m, sigma)
        noise = (noise > 0) * noise - (noise > 255) * (noise - 255)
        noise = noise.transpose((2, 0, 1))
        srcA = srcA.transpose((2, 0, 1))
        srcA[0:3] = np.multiply(srcA[0:3], (1 - noise[3] / 255)) + np.multiply(noise[0:3], noise[3] / 255)
        srcA[3] = np.multiply(srcA[3], (1 - noise[3] / 255)) + noise[3]
        srcA = srcA.transpose((1, 2, 0))

    if rand() < 0.6 * strength_mod * DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]:
        strength = strength_mod * DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]
        rad = (rows + cols) / 4 * (0.25 + 0.75 * rand())
        thickness = (rows + cols) / 4 * strength * rand()
        cv2.circle(srcA, (int(rows * rand()), int(cols * rand())), int(rad + thickness/2),
                   (0, 0, 0, 0), int(thickness))

    if rand() < 0.6 * strength_mod * DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]:
        strength = strength_mod * DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]
        for j in range(int(rand() * rand() * 5 * strength) + 1):
            x = [(int(rows * (-0.5 + 2 * rand())), int(cols * (-0.5 + 2 * rand()))) for i in range(2)]
            cv2.line(srcA, x[0], x[1], (0, 0, 0, 0), int(1 + rand() * (rows + cols) / 10 * strength))

    if rand() < 0.6 * strength_mod * DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]:
        strength = strength_mod * DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]
        text = np.zeros((rows, cols, 4))
        for j in range(int(1 + 5 * rand() * strength)):
            txt = "".join([random.choice(string.ascii_letters) for i in range(int(15 * rand(0.1, strength)))])
            cv2.putText(text, txt, (int(rows * rand()), int(cols *rand())),
                        cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1 + rand() * 5,
                        (int(rand() * 255), int(rand() * 255), int(rand() * 255), 255),
                        int(5 + rand() * 10))
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 360 * rand(), 1)
        text = cv2.warpAffine(text, M, (cols, rows))
        text = text.transpose((2, 0, 1))
        srcA = srcA.transpose((2, 0, 1))
        srcA[0:3] = np.multiply(srcA[0:3], (1 - text[3] / 255)) + np.multiply(text[0:3], text[3] / 255)
        srcA[3] = np.multiply(srcA[3], (1 - text[3] / 255)) + text[3]
        srcA = srcA.transpose((1, 2, 0))

    if rand() < 0.25 * strength_mod * DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]:
        strength = strength_mod * DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]
        srcA = cv2.medianBlur(srcA, 1 + 2 * int(rand(0, strength)**3 * rand(0, strength) * (rows + cols) / 40))

    if rand() < 0.25 * strength_mod * DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]:
        strength = strength_mod * DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]
        size_1 = 1 + 2 * int(rand(0, strength)**3 * rand(0, strength) * (rows + cols) / 200)
        size_2 = 1 + 2 * int(rand(0, strength)**3 * rand(0, strength) * (rows + cols) / 200)
        srcA = cv2.blur(srcA, (size_1, size_2))

    if rand() < 0.3 * strength_mod * DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]:
        strength = strength_mod * DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]
        srcA = srcA.transpose((2, 0, 1))
        c = 1. - 0.6 * rand(0, strength)
        a = np.linspace(c + rand(0, strength) * (1 - c), c + rand(0, strength) * (1 - c), rows)[:, None]
        b = np.linspace(c + rand(0, strength) * (1 - c), c + rand(0, strength) * (1 - c), cols)[None, :]
        gradient = np.multiply(a, b)
        srcA[0:3] = 255 - np.multiply(255 - srcA[0:3], gradient)
        srcA = srcA.transpose((1, 2, 0))

    if rand() < 0.5 * strength_mod * DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]:
        strength = strength_mod * DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]
        srcA = srcA.transpose((2, 0, 1))
        c = 1. - 0.6 * rand(0, strength)
        a = np.linspace(c + rand(0, strength) * (1 - c), c + rand(0, strength) * (1 - c), rows)[:, None]
        b = np.linspace(c + rand(0, strength) * (1 - c), c + rand(0, strength) * (1 - c), cols)[None, :]
        gradient = np.multiply(a, b)
        srcA[0:3] = np.multiply(srcA[0:3], gradient)
        srcA = srcA.transpose((1, 2, 0))

    if rand() < 0.95 * strength_mod * DATASET_OBJECT_COLOUR_FILTER_STRENGTH[config_key]:
        strength = strength_mod * DATASET_OBJECT_COLOUR_FILTER_STRENGTH[config_key]
        hue = 0.3 * strength
        sat = 1 + 1. * strength
        val = 1 + 1. * strength
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        src = srcA[:, :, :3]
        x = rgb_to_hsv(src / 255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        src = hsv_to_rgb(x) * 255
        srcA[:, :, 0:3] = src

    return srcA, name


def objectPerspective(srcA, name, size_step):
    strength_mod = DATASET_FILTERING_STRENGTHS[size_step]
    config_key = name_to_config_key(name)
    names = name.split(';')
    rows, cols, colors = srcA.shape
    if len(names) == 4:
        points = [np.float32([[rows // 2, cols // 2], [rows, cols // 2],
                             [rows // 2, cols], [rows, cols]]),
                  np.float32([[rows, cols // 2], [rows // 2 + rows, cols // 2],
                             [rows, cols], [rows // 2 + rows, cols]]),
                  np.float32([[rows // 2, cols], [rows, cols],
                             [rows // 2, cols // 2 + cols], [rows, cols // 2 + cols]]),
                  np.float32([[rows, cols], [rows // 2 + rows, cols],
                             [rows, cols // 2 + cols], [rows // 2 + rows, cols // 2 + cols]])]
    else:
        points = [np.float32([[cols // 2, rows // 2],
                             [cols // 2 + cols, rows // 2],
                             [cols // 2, rows // 2 + rows],
                             [cols // 2 + cols, rows // 2 + rows]])]

    corners = []
    img = np.zeros((rows * 2, cols * 2, 4))
    img[rows // 2:rows // 2 + rows,cols // 2:cols // 2 + cols] = srcA

    # Warp

    # Perspective
    a = 1/2 - 0.3 * strength_mod * DATASET_OBJECT_PERSPECTIVE_STRENGTH[config_key]
    pts1 = np.float32([[rows // 2, cols // 2], [rows // 2 + rows, cols // 2],
                       [rows // 2, cols // 2 + cols], [rows // 2 + rows, cols // 2 + cols]])
    pts2 = np.float32([[int(rows * (a + random.random() * (1 - 2 * a))),
                        int(cols * (a + random.random() * (1 - 2 * a)))],
                       [int(rows * (a + random.random() * (1 - 2 * a))) + rows,
                        int(cols * (a + random.random() * (1 - 2 * a)))],
                       [int(rows * (a + random.random() * (1 - 2 * a))),
                        int(cols * (a + random.random() * (1 - 2 * a))) + cols],
                       [int(rows * (a + random.random() * (1 - 2 * a))) + rows,
                        int(cols * (a + random.random() * (1 - 2 * a))) + cols]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img,M,(rows * 2, cols * 2))
    for a in range(len(points)):
        points[a] = np.float32( cv2.perspectiveTransform(points[a].reshape(1, -1, 2), M).reshape(-1, 2))

    # Rotation 1 / 2
    M = cv2.getRotationMatrix2D((cols, rows), (1 - 2 * rand()) * 180 * DATASET_OBJECT_ROTATION_STRENGTH[config_key] / 2, 1)
    img = cv2.warpAffine(img, M, (rows * 2, cols * 2))
    for a in range(len(points)):
        points[a] = np.int32( cv2.transform(points[a].reshape(1, -1, 2), M).reshape(-1, 2))

    # Camera distortion
    DIM = img.shape[:2]
    K = np.array([[img.shape[0] * rand(0.4, 0.6), 0.0, img.shape[0] * rand(0.4, 0.6)],
                  [0.0, img.shape[0] * rand(0.4, 0.6), img.shape[0] * rand(0.4, 0.6)],
                  [0.0, 0.0, 1.0]])
    if rand(-1, 1) > 0:
        center = 3 * DATASET_OBJECT_DISTORT_STRENGTH[config_key] * strength_mod
        scale = 4 * DATASET_OBJECT_DISTORT_STRENGTH[config_key] * strength_mod
    else:
        center = -0.2 * DATASET_OBJECT_DISTORT_STRENGTH[config_key] * strength_mod
        scale = 0.5 * DATASET_OBJECT_DISTORT_STRENGTH[config_key] * strength_mod
    D = np.array([[0.707107 + center + scale * rand(-1,1)], [center + scale * rand(-1,1)],
                  [0.707107 + center + scale * rand(-1,1)], [center + scale * rand(-1,1)]])

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Rotation 2 / 2
    M = cv2.getRotationMatrix2D((cols, rows), (1 - 2 * rand()) * 180 * DATASET_OBJECT_ROTATION_STRENGTH[config_key] / 2, 1)
    img = cv2.warpAffine(img, M, (rows * 2, cols * 2))
    for a in range(len(points)):
        points[a] = np.int32( cv2.transform(points[a].reshape(1, -1, 2), M).reshape(-1, 2))

    for a in range(len(points)):
        corners.append([(np.min(points[a].transpose()[0]), np.min(points[a].transpose()[1])),
                        (np.max(points[a].transpose()[0]), np.max(points[a].transpose()[1]))])

    # Erode alpha
    img[:,:,3] = np.round(img[:,:,3] / 255).astype(img.dtype) * 255
    erosion_size = int(rand(1, DATASET_EROSION_MAX_SIZE[size_step]))
    erosion_type = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(erosion_type, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
    img[:,:,3] = cv2.erode(img[:,:,3], element)
    # diff_alpha = img[:,:,3] - new_alpha
    # print(np.sum(diff_alpha), np.sum(diff_alpha) / np.sum(img[:,:,3]))
    # img = img.astype(np.int16)
    # img = img.transpose((2, 0, 1))
    # img += diff_alpha.astype(np.int16)
    # img = np.clip(img, 0, 255).astype(np.uint8)
    # img = img.transpose((1, 2, 0))

    namesPositions = []
    for i in range(len(names)):
        namesPositions.append([names[i]])
        namesPositions[-1].append(corners[i][0][0])
        namesPositions[-1].append(corners[i][0][1])
        namesPositions[-1].append(corners[i][1][0])
        namesPositions[-1].append(corners[i][1][1])

    return img, namesPositions


def makeBase(base_num, size_step):
    base_img = cv2.imread(baseFiles[base_num], cv2.IMREAD_COLOR)
    base_name_pos = copy.deepcopy(baseDefaultNamesPositions[base_num])

    scale_h = DATASET_TRAINING_SHAPES[size_step][0] / base_img.shape[0]
    scale_w = DATASET_TRAINING_SHAPES[size_step][1] / base_img.shape[1]
    scale = max(scale_h, scale_w)
    new_shape = tuple(map(int, np.round([base_img.shape[0] * scale, base_img.shape[1] * scale])))

    base_img = cv2.resize(base_img, new_shape[::-1])

    for name_pos in base_name_pos:
        for i in range(1, 5):
            name_pos[i] = int(scale * name_pos[i])

    position = (random.randint(0, new_shape[0] - DATASET_TRAINING_SHAPES[size_step][0]),
                random.randint(0, new_shape[1] - DATASET_TRAINING_SHAPES[size_step][1]))
    base_img = base_img[position[0]:position[0] + DATASET_TRAINING_SHAPES[size_step][0],
                        position[1]:position[1] + DATASET_TRAINING_SHAPES[size_step][1]]

    final_name_pos = []
    for name_pos in base_name_pos:
        for i in range(1, 5):
            name_pos[i] = min(max(name_pos[i] - position[i % 2], 0), DATASET_TRAINING_SHAPES[size_step][i % 2])
        if (name_pos[3] - name_pos[1]) * (name_pos[4] - name_pos[2]) > 0:
            final_name_pos.append(name_pos)

    return base_img, final_name_pos


def placeObject(obj, objectNamesPositions, base, baseNamesPositions, size_step):
    strength_mod = DATASET_FILTERING_STRENGTHS[size_step]
    for _ in range(10):
        config_key = name_to_config_key(objectNamesPositions[0][0])
        size_max = 0.3 + 2 / (len(objectNamesPositions) + len(baseNamesPositions))
        size = 0.02 + size_max * (1 - strength_mod * DATASET_OBJECT_ZOOM_STRENGTH[config_key]) + \
               size_max * rand() * rand() * strength_mod * DATASET_OBJECT_ZOOM_STRENGTH[config_key]
        label = cv2.resize(obj, None, fx=size, fy=size, interpolation=cv2.INTER_AREA)
        lrows, lcols, lcolors = label.shape
        brows, bcols, bcolors = base.shape

        if len(objectNamesPositions) == 4:
            position = (int(random.random() * (brows - lrows)), int(random.random() * (bcols - lcols)))
        else:
            position = (int(random.random() * (brows - lrows * (1 - strength_mod * DATASET_OBJECT_CROP_STRENGTH[config_key]))
                            - lrows / 2 * strength_mod * DATASET_OBJECT_CROP_STRENGTH[config_key]),
                        int(random.random() * (bcols - lcols * (1 - strength_mod * DATASET_OBJECT_CROP_STRENGTH[config_key]))
                            - lcols / 2 * strength_mod * DATASET_OBJECT_CROP_STRENGTH[config_key]))

        overlapped = False
        for base_name_pos in baseNamesPositions:
            rect_b = np.asarray(base_name_pos[1:])
            rect_f = np.asarray([position[0], position[1], position[0] + lrows, position[1] + lcols])
            rect_i = np.concatenate((np.maximum(rect_b[:2], rect_f[:2]), np.minimum(rect_b[2:], rect_f[2:])))
            if (rect_i[2] - rect_i[0]) * (rect_i[3] - rect_i[1]) > \
                    0.8 * (rect_b[2] - rect_b[0]) * (rect_b[3] - rect_b[1]):
                overlapped = True
                break
        if overlapped:
            continue

        label = label.transpose((2, 0, 1))
        base = base.transpose((2, 0, 1))

        base[0:3, max(0, position[0]):min(brows, position[0] + lrows),
                             max(0, position[1]):min(bcols, position[1] + lcols)] =\
            np.multiply(base[0:3, max(0, position[0]):min(brows, position[0] + lrows),
                             max(0, position[1]):min(bcols, position[1] + lcols)],
                        (1 - label[3, max(0, -position[0]):min(lrows, brows - position[0]),
                                   max(0, -position[1]):min(lcols, bcols - position[1])] / 255))\
            + np.multiply(label[0:3, max(0, -position[0]):min(lrows, brows - position[0]),
                                   max(0, -position[1]):min(lcols, bcols - position[1])],
                          label[3, max(0, -position[0]):min(lrows, brows - position[0]),
                                   max(0, -position[1]):min(lcols, bcols - position[1])] / 255)
        base = base.transpose((1, 2, 0))

        for a in range(len(objectNamesPositions)):
            for b in range(2):
                objectNamesPositions[a][1 + 2 * b] = max(min(round(int(objectNamesPositions[a][1 + 2 * b])*size) + position[1], bcols), 0)
                objectNamesPositions[a][2 + 2 * b] = max(min(round(int(objectNamesPositions[a][2 + 2 * b])*size) + position[0], brows), 0)
        baseNamesPositions.extend(objectNamesPositions)
        break
    return base, baseNamesPositions


def filterImages(base, namesPositions, size_step):
    rows, cols, channels = base.shape[:3]
    base = np.asarray(base, np.uint8)
    strength_mod = DATASET_FILTERING_STRENGTHS[size_step]

    keys = []
    for key in DATASET_OBJECT_CROP_STRENGTH.keys():
        for namePos in namesPositions:
            if key in namePos[0]:
                keys.append(key)
                break
    keys = list(set(keys))

    if len(keys) > 0:
        blur_strength = np.min([strength_mod * DATASET_OBJECT_BLUR_CUT_STRENGTH[key] for key in keys])
        colour_strength = np.min([strength_mod * DATASET_OBJECT_COLOUR_FILTER_STRENGTH[key] for key in keys])
    else:
        blur_strength = 1.
        colour_strength = 1.

    if random.random() < 0.5 * strength_mod:
        angle = 2 * np.pi * rand()
        x = np.sin(angle)
        y = np.cos(angle)
        src = base
        reps = int(rand() * rand() * 100 * strength_mod)
        step = (rand(0.02, 0.3) + rand(0, 0.3)) * strength_mod
        base = base / (2 * reps + 1)

        for i in range(reps):
            M1 = np.float32([[1, 0, round(x * (i + 1) * step)], [0, 1, round(y * (i + 1) * step)]])
            M2 = np.float32([[1, 0, -round(x * (i + 1) * step)], [0, 1, -round(y * (i + 1) * step)]])
            base += (cv2.warpAffine(src, M1, (cols, rows)) / (2 * reps + 1)
                     + cv2.warpAffine(src, M2, (cols, rows)) / (2 * reps + 1))
        base = np.asarray(base, np.uint8)

    if random.random() < 0.2 * blur_strength:
        base = cv2.medianBlur(base, 1 + 2 * int(rand(0, blur_strength)**3 * rand(0, blur_strength) * (rows + cols) / 100))

    if random.random() < 0.2 * blur_strength:
        size_1 = 1 + 2 * int(rand(0, blur_strength)**3 * rand(0, blur_strength) * (rows + cols) / 800)
        size_2 = 1 + 2 * int(rand(0, blur_strength)**3 * rand(0, blur_strength) * (rows + cols) / 800)
        base = cv2.blur(base, (size_1, size_2))

    if random.random() < 0.5 * colour_strength:
        hue = 0.05 * colour_strength
        sat = 1 + 1. * colour_strength
        val = 1 + 1. * colour_strength
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = rgb_to_hsv(base / 255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        base = hsv_to_rgb(x) * 255

    return base, namesPositions


def writeImages(prefix, index, annotations_q, base, namesPositions, size_step):
    strength_mod = DATASET_FILTERING_STRENGTHS[size_step]
    dataset_f = DATASET_LOCATION + str(size_step)

    is_jpg = rand() < 0.5 * strength_mod

    if DEBUG:
        print(namesPositions)
        if(len(namesPositions) == 0):
            print("!!!!!!!!!!!!!!!!!!!!!!!!")
        for a in range(len(namesPositions)):
            print("DRAWING RECTANGLE!!!")
            cv2.rectangle(base, tuple(namesPositions[a][1:3]), tuple(namesPositions[a][3:]), (255, 0, 255, 255), 1)

    file_name = prefix + str(index).zfill(7) + (".jpg " if is_jpg else ".png ")
    ann_line = os.getcwd() + "/" + dataset_f + "/" + file_name
    for a in range(len(namesPositions)):
        if objectNums.get(namesPositions[a][0]) is None:
            objectNums[namesPositions[a][0]] = 1
        else:
            objectNums[namesPositions[a][0]] += 1
        if (namesPositions[a][1] - namesPositions[a][3]) * (namesPositions[a][2] - namesPositions[a][4]) == 0:
            continue
        ann_line += ','.join(map(str, namesPositions[a][1:]))
        ann_line += ',' + str(objectIDs[namesPositions[a][0]])
        if a != len(namesPositions) - 1:
            ann_line += ' '
    ann_line += '\n'
    # print("saving as", dataset_f + "/" + str(index).zfill(7) + ".png")
    if is_jpg:
        cv2.imwrite(dataset_f + "/" + file_name, base, [cv2.IMWRITE_JPEG_QUALITY, int(rand(85,10))])
    else:
        cv2.imwrite(dataset_f + "/" + file_name, base)
    annotations_q.put(ann_line)


def createLabel(base_num, obj_nums, size_step):
    base_img, base_name_pos = makeBase(base_num, size_step)
    for obj_num in obj_nums:
        obj_img, obj_name_pos = objectMake(obj_num, size_step)
        obj_img, obj_name_pos = objectFilter(obj_img, obj_name_pos, size_step)
        obj_img, obj_name_pos = objectPerspective(obj_img, obj_name_pos, size_step)
        base_img, base_name_pos = placeObject(obj_img, obj_name_pos, base_img, base_name_pos, size_step)
    base_img, base_name_pos = filterImages(base_img, base_name_pos, size_step)
    return base_img, base_name_pos


def threadedCreateLabels(prefix, number, size_step, annotations_q):
    print("threadedCreateLabels")
    global objectList, objectImgs, objectNames, objectIDs, baseList, baseFiles, baseDefaultNamesPositions, objectNums, \
        objectTree
    dataset_f = DATASET_LOCATION + str(size_step)

    objectList = []
    objectImgs = []
    objectNames = []
    objectIDs = {}
    baseList = []
    objectNums = {}
    objectTree = None

    makeObjectList(size_step)
    makeBaseList(size_step)
    for k in range(number):
        num_labels = np.random.choice(list(range(len(DATASET_OBJECT_PLACE_CHANCE))),
                                      p=DATASET_OBJECT_PLACE_CHANCE)

        base_num = baseList.pop(0)
        obj_nums = [objectList.pop(0) for i in range(num_labels)]

        base_img, base_name_pos = createLabel(base_num, obj_nums, size_step)

        print(".", end="")
        sys.stdout.flush()
        if random.random() < 0.01:
            print(len(os.listdir(dataset_f)) - 2)

        writeImages(prefix, k, annotations_q, base_img, base_name_pos, size_step)

    print("threadedCreateLabels finished")
    return True


def createLabels():
    global objectList, objectImgs, objectNames, objectIDs, baseList, baseFiles, baseDefaultNamesPositions, objectNums, \
        objectTree
    num_threads = DATASET_CREATION_THREADS
    for size_step in range(DATASET_SIZE_STEPS):
        dataset_f = DATASET_LOCATION + str(size_step)
        if not os.path.exists(dataset_f) or len(os.listdir(dataset_f)) < DATASET_NUM_IMAGES \
                or REBUILD_DATASET:

            if os.path.exists(dataset_f):
                shutil.rmtree(dataset_f)
                time.sleep(0.2)
            os.makedirs(dataset_f)

            print(size_step)

            filterObjects(size_step)

            # ann_q = mp.Queue()
            m = mp.Manager()
            ann_q = m.Queue()
            ann_f = open(dataset_f + "/labels.txt", "a+")

            params = [(str(thread_i).zfill(4) + "_", DATASET_NUM_IMAGES // num_threads +
                      (1 if DATASET_NUM_IMAGES % num_threads > thread_i else 0), size_step, ann_q)
                      for thread_i in range(num_threads)]
            pool = mp.Pool(processes=num_threads)

            print(params)
            result = pool.starmap_async(threadedCreateLabels, params)

            t = time.time() - 300
            while not result.ready() or not ann_q.empty():
                try:
                    ann = ann_q.get_nowait()
                    ann_f.write(ann)
                except qEmpty:
                    pass
                if time.time() - t > 300:
                    t = time.time()
                    # gives a single float value
                    print("cpu", psutil.cpu_percent(interval=None, percpu=True))
                    # gives an object with many fields
                    print("memory", psutil.virtual_memory())
            print(result.get())
            time.sleep(0.1)
            while not ann_q.empty():
                try:
                    ann = ann_q.get_nowait()
                    ann_f.write(ann)
                except qEmpty:
                    pass
            pool.close()
            pool.join()
            ann_f.close()


def createDataset(debug=False):
    global DEBUG, DATASET_NUM_IMAGES
    if debug:
        DEBUG = True
        DATASET_NUM_IMAGES = 50
    print(DATASET_NUM_IMAGES)
    if not os.path.exists("openimages/test") or not os.path.exists("openimages/validation") or REDOWNLOAD_DATASET:
        newDownload()
    if not os.path.exists("openimages/all-annotations-bbox.csv") or REFILTER_DATASET:
        filterOpenImages()
    createLabels()


if __name__=="__main__":
    createDataset()
