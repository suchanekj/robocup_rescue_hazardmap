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


from config import *


objectList = []
objectImgs = []
objectNames = []
objectIDs = {}
baseList = []
baseFiles = []
baseDefaultNamesPositions = []


# Door - /m/02dgv
# Fire extinguisher - folder
# Baby doll - folder
# Person - /m/03bt1vf /m/04yx4 /m/01g317
# Valve - floder
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


def filterObjects():
    object_fs = os.listdir("objects")
    shutil.rmtree("filtered_objects/", ignore_errors=True)
    time.sleep(0.5)
    os.makedirs("filtered_objects/")
    for obj in object_fs:
        filtering = (0, 0)
        for key in DATASET_OBJECT_BACKGROUND_REMOVAL.keys():
            if key in obj:
                filtering = DATASET_OBJECT_BACKGROUND_REMOVAL[key]
        if os.path.isdir("objects/" + obj):
            os.makedirs("filtered_objects/" + obj)
            objs = ["objects/" + obj + "/" + f for f in os.listdir("objects/" + obj)]
        else:
            objs = ["objects/" + obj]
        for f in objs:
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

                res = cv2.resize(res, (400, 400))

            else:
                res = img[x0:x1, y0:y1]

                scale_h = 400 / res.shape[0]
                scale_w = 400 / res.shape[0]
                scale = min(scale_h, scale_w)
                new_shape = tuple(map(int, np.round([res.shape[1] * scale, res.shape[0] * scale])))

                res = cv2.resize(res, new_shape)

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


def makeObjectList():
    global objectImgs
    global objectList
    global objectNames
    global objectIDs
    object_fs = os.listdir("filtered_objects")
    names_f = open(DATASET_LOCATION + "/labelNames.txt", "w")
    from_bases_num = len(objectNames)
    for obj in object_fs:
        objectImgs.append([])
        name = obj.split(".")[0]
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
    for i in range(int(DATASET_MAX_OBJECTS_PER_IMG * DATASET_NUM_IMAGES / len(objectImgs) *
                       (DATASET_FOURS_PART * 3 + 1)) * 2):
        all_objs = list(range(from_bases_num, len(objectNames)))
        hazard_labels = [obj for obj in all_objs if "hazmat" in objectNames[obj]]
        all_objs = [obj for obj in all_objs if "hazmat" not in objectNames[obj]]
        random.shuffle(hazard_labels)
        while len(hazard_labels) >= 4:
            if rand() < DATASET_FOURS_PART * ((len(hazard_labels) / 4) / (len(hazard_labels) // 4)):
                all_objs.append(hazard_labels[:4])
                hazard_labels = hazard_labels[4:]
            else:
                all_objs.append(hazard_labels[0])
                hazard_labels = hazard_labels[1:]
        all_objs.extend(hazard_labels)
        random.shuffle(all_objs)
        objectList.extend(all_objs)
    names_f.close()


def makeBaseList():
    global baseList
    global baseFiles
    global baseDefaultNamesPositions
    names_f = open(DATASET_LOCATION + "/labelNames.txt", "a")
    doorBases = []
    personBases = []
    nothingBases = []
    parts = ["validation", "test"]
    all_df = pd.read_csv("openimages/all-annotations-bbox.csv", index_col=0)
    doors = 0
    persons = 0
    objectSplit = np.sum(np.multiply(DATASET_OBJECT_PLACE_CHANCE, range(DATASET_MAX_OBJECTS_PER_IMG + 1))) / \
                  len(objectNames) * 1.2

    objectFromBaseNames = list(set(DATASET_OPENIMAGES_LABEL_TO_OBJECT.values()))
    objectNames.extend(objectFromBaseNames)
    for objName in objectFromBaseNames:
        objectIDs[objName] = len(objectIDs.keys())
        names_f.write(str(objectIDs[objName]) + " " + objName + "\n")

    id = 0
    for part in parts:
        files = os.listdir("openimages/" + part)
        for f in files:
            if len(doorBases) >= DATASET_NUM_IMAGES // 20:
                continue
            name = f[:-4]
            objects_df = all_df[all_df["ImageID"] == name]
            if len(objects_df) > 3:
                continue
            print(".", end="")
            sys.stdout.flush()
            if (id + 1) % 100 == 0:
                print(id)
            if len(objects_df) == 0:
                nothingBases.append(id)
            elif DATASET_OPENIMAGES_LABEL_TO_OBJECT[objects_df["LabelName"].values[0]] == "door":
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

    print("loaded, making list", objectSplit)
    for i in range(DATASET_NUM_IMAGES * 2):
        print(".", end="")
        if (i + 1) % 100 == 0:
            print(i)
        if doors < i * objectSplit:
            chosen_base = random.choice(doorBases)
        elif persons < i * objectSplit:
            chosen_base = random.choice(personBases)
        else:
            chosen_base = random.choice(nothingBases)
        for namePosition in baseDefaultNamesPositions[chosen_base]:
            if namePosition[0] == "door":
                doors += 1
            if namePosition[0] == "person":
                persons += 1
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


def objectMake(obj):
    out = [None, None]
    if isinstance(obj, Iterable):
        crop_strength = DATASET_OBJECT_CROP_STRENGTH["hazmat"]
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
        crop_strength = DATASET_OBJECT_CROP_STRENGTH[name_to_config_key(out[1])]
        crop_x = rand(0, crop_strength) if rand() < (0.5 + crop_strength) else 0
        img = random.choice(objectImgs[obj])
        img = copy.deepcopy(img)
        out[0] = crop(img, crop_x)

    return out


def objectFilter(srcA, name):
    rows, cols, colors = srcA.shape
    srcA = np.asarray(srcA, np.uint8)

    config_key = name_to_config_key(name)

    if rand() < 0.9 * DATASET_OBJECT_COLOUR_FILTER_STRENGTH[config_key]:
        strength = DATASET_OBJECT_COLOUR_FILTER_STRENGTH[config_key]
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

    if rand() < 0.6 * DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]:
        strength = DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]
        rad = (rows + cols) / 4 * (0.25 + 0.75 * rand())
        thickness = (rows + cols) / 4 * strength * rand()
        cv2.circle(srcA, (int(rows * rand()), int(cols * rand())), int(rad + thickness/2),
                   (0, 0, 0, 0), int(thickness))

    if rand() < 0.6 * DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]:
        strength = DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]
        for j in range(int(rand() * rand() * 5 * strength) + 1):
            x = [(int(rows * (-0.5 + 2 * rand())), int(cols * (-0.5 + 2 * rand()))) for i in range(2)]
            cv2.line(srcA, x[0], x[1], (0, 0, 0, 0), int(1 + rand() * (rows + cols) / 10 * strength))

    if rand() < 0.6 * DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]:
        strength = DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]
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

    if rand() < 0.25 * DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]:
        strength = DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]
        srcA = cv2.medianBlur(srcA, 1 + 2 * int(rand(0, strength)**3 * rand(0, strength) * (rows + cols) / 40))

    if rand() < 0.25 * DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]:
        strength = DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]
        size_1 = 1 + 2 * int(rand(0, strength)**3 * rand(0, strength) * (rows + cols) / 200)
        size_2 = 1 + 2 * int(rand(0, strength)**3 * rand(0, strength) * (rows + cols) / 200)
        srcA = cv2.blur(srcA, (size_1, size_2))

    if rand() < 0.3 * DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]:
        strength = DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]
        srcA = srcA.transpose((2, 0, 1))
        c = 1. - 0.6 * rand(0, strength)
        a = np.linspace(c + rand(0, strength) * (1 - c), c + rand(0, strength) * (1 - c), rows)[:, None]
        b = np.linspace(c + rand(0, strength) * (1 - c), c + rand(0, strength) * (1 - c), cols)[None, :]
        gradient = np.multiply(a, b)
        srcA[0:3] = 255 - np.multiply(255 - srcA[0:3], gradient)
        srcA = srcA.transpose((1, 2, 0))

    if rand() < 0.5 * DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]:
        strength = DATASET_OBJECT_BLUR_CUT_STRENGTH[config_key]
        srcA = srcA.transpose((2, 0, 1))
        c = 1. - 0.6 * rand(0, strength)
        a = np.linspace(c + rand(0, strength) * (1 - c), c + rand(0, strength) * (1 - c), rows)[:, None]
        b = np.linspace(c + rand(0, strength) * (1 - c), c + rand(0, strength) * (1 - c), cols)[None, :]
        gradient = np.multiply(a, b)
        srcA[0:3] = np.multiply(srcA[0:3], gradient)
        srcA = srcA.transpose((1, 2, 0))

    if rand() < 0.95 * DATASET_OBJECT_COLOUR_FILTER_STRENGTH[config_key]:
        strength = DATASET_OBJECT_COLOUR_FILTER_STRENGTH[config_key]
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


def objectPerspective(srcA, name):
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

    a = 1/2 - 0.3 * DATASET_OBJECT_PERSPECTIVE_STRENGTH[config_key]
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

    M = cv2.getRotationMatrix2D((cols, rows), rand(), 1)
    img = cv2.warpAffine(img, M, (rows * 2, cols * 2))
    for a in range(len(points)):
        points[a] = np.int32( cv2.transform(points[a].reshape(1, -1, 2), M).reshape(-1, 2))
        corners.append([(np.min(points[a].transpose()[0]), np.min(points[a].transpose()[1])),
                        (np.max(points[a].transpose()[0]), np.max(points[a].transpose()[1]))])

    namesPositions = []
    for i in range(len(names)):
        namesPositions.append([names[i]])
        namesPositions[-1].append(corners[i][0][0])
        namesPositions[-1].append(corners[i][0][1])
        namesPositions[-1].append(corners[i][1][0])
        namesPositions[-1].append(corners[i][1][1])

    return img, namesPositions


def makeBase(base_num):
    base_img = cv2.imread(baseFiles[base_num], cv2.IMREAD_COLOR)
    base_name_pos = copy.deepcopy(baseDefaultNamesPositions[base_num])

    scale_h = DATASET_DEFAULT_SHAPE[0] / base_img.shape[0]
    scale_w = DATASET_DEFAULT_SHAPE[1] / base_img.shape[1]
    scale = max(scale_h, scale_w)
    new_shape = tuple(map(int, np.round([base_img.shape[0] * scale, base_img.shape[1] * scale])))

    base_img = cv2.resize(base_img, new_shape[::-1])

    for name_pos in base_name_pos:
        for i in range(1, 5):
            name_pos[i] = int(scale * name_pos[i])

    print(new_shape, base_img.shape, DATASET_DEFAULT_SHAPE)

    position = (random.randint(0, new_shape[0] - DATASET_DEFAULT_SHAPE[0]),
                random.randint(0, new_shape[1] - DATASET_DEFAULT_SHAPE[1]))
    base_img = base_img[position[0]:position[0] + DATASET_DEFAULT_SHAPE[0],
                        position[1]:position[1] + DATASET_DEFAULT_SHAPE[1]]

    final_name_pos = []
    for name_pos in base_name_pos:
        for i in range(1, 5):
            name_pos[i] = min(max(name_pos[i] - position[i % 2], 0), DATASET_DEFAULT_SHAPE[i % 2])
        if (name_pos[3] - name_pos[1]) * (name_pos[4] - name_pos[2]) > 0:
            final_name_pos.append(name_pos)

    return base_img, final_name_pos


def placeObject(obj, objectNamesPositions, base, baseNamesPositions):
    for _ in range(10):
        config_key = name_to_config_key(objectNamesPositions[0][0])
        size_max = 0.3 + 2 / (len(objectNamesPositions) + len(baseNamesPositions))
        size = 0.02 + size_max * (1 - DATASET_OBJECT_ZOOM_STRENGTH[config_key]) + \
               size_max * rand() * rand() * DATASET_OBJECT_ZOOM_STRENGTH[config_key]
        label = cv2.resize(obj, None, fx=size, fy=size, interpolation=cv2.INTER_AREA)
        lrows, lcols, lcolors = label.shape
        brows, bcols, bcolors = base.shape

        if len(objectNamesPositions) == 4:
            position = (int(random.random() * (brows - lrows)), int(random.random() * (bcols - lcols)))
        else:
            position = (int(random.random() * (brows - lrows * (1 - DATASET_OBJECT_CROP_STRENGTH[config_key]))
                            - lrows / 2 * DATASET_OBJECT_CROP_STRENGTH[config_key]),
                        int(random.random() * (bcols - lcols * (1 - DATASET_OBJECT_CROP_STRENGTH[config_key]))
                            - lcols / 2 * DATASET_OBJECT_CROP_STRENGTH[config_key]))

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


def filterImages(base, namesPositions):
    rows, cols, channels = base.shape[:3]
    base = np.asarray(base, np.uint8)

    keys = []
    for key in DATASET_OBJECT_CROP_STRENGTH.keys():
        for namePos in namesPositions:
            if key in namePos[0]:
                keys.append(key)
                break
    keys = list(set(keys))

    if len(keys) > 0:
        blur_strength = np.min([DATASET_OBJECT_BLUR_CUT_STRENGTH[key] for key in keys])
        colour_strength = np.min([DATASET_OBJECT_COLOUR_FILTER_STRENGTH[key] for key in keys])
    else:
        blur_strength = 1.
        colour_strength = 1.

    if random.random() < 0.5 * blur_strength:
        angle = 2 * np.pi * rand()
        x = np.sin(angle)
        y = np.cos(angle)
        src = base
        reps = int(rand() * rand() * 50 * blur_strength)
        step = (rand(0.02, 0.3) + rand(0, 0.3)) * blur_strength
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
        hue = 0.3 * colour_strength
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


def writeImages(base, namesPositions):
    f = open(DATASET_LOCATION + "/labels.txt", "a+")

    for a in range(len(namesPositions)):
        cv2.rectangle(base, tuple(namesPositions[a][1:3]), tuple(namesPositions[a][3:]), (255, 0, 255, 255), 1)

    index = len(os.listdir(DATASET_LOCATION)) - 2
    f.write(os.getcwd() + "\\" + DATASET_LOCATION + "\\" + str(index).zfill(7) + ".png ")
    for a in range(len(namesPositions)):
        if (namesPositions[a][1] - namesPositions[a][3]) * (namesPositions[a][2] - namesPositions[a][3]) == 0:
            continue
        f.write(','.join(map(str, namesPositions[a][1:])))
        f.write(',')
        f.write(str(objectIDs[namesPositions[a][0]]))
        if a != len(namesPositions) - 1:
            f.write(' ')
    f.write('\n')
    cv2.imwrite(DATASET_LOCATION + "\\" + str(index).zfill(7) + ".png", base)
    f.close()


def createLabels():
    if os.path.exists(DATASET_LOCATION):
        shutil.rmtree(DATASET_LOCATION)
        time.sleep(0.2)
    os.makedirs(DATASET_LOCATION)
    print("make object list")
    filterObjects()
    makeObjectList()
    print("make_base_list")
    makeBaseList()
    print("make dataset")
    time_makeBase = 0
    time_objectMake = 0
    time_objectFilter = 0
    time_objectPerspective = 0
    time_placeObject = 0
    time_filterImages = 0
    time_writeImages = 0
    while len(os.listdir(DATASET_LOCATION)) - 2 < DATASET_NUM_IMAGES:
        print(".", end="")
        sys.stdout.flush()
        if len(os.listdir(DATASET_LOCATION)) % 100 == 0:
            print(len(os.listdir(DATASET_LOCATION)) - 2)
        num_labels = np.random.choice(list(range(len(DATASET_OBJECT_PLACE_CHANCE))), p=DATASET_OBJECT_PLACE_CHANCE)

        t = time.time()

        base_num = baseList.pop(0)
        base_img, base_name_pos = makeBase(base_num)
        time_makeBase += time.time() - t
        t = time.time()

        for i in range(num_labels):
            obj_num = objectList.pop(0)
            obj_img, obj_name_pos = objectMake(obj_num)
            time_objectMake += time.time() - t
            t = time.time()
            obj_img, obj_name_pos = objectFilter(obj_img, obj_name_pos)
            time_objectFilter += time.time() - t
            t = time.time()
            obj_img, obj_name_pos = objectPerspective(obj_img, obj_name_pos)
            time_objectPerspective += time.time() - t
            t = time.time()
            base_img, base_name_pos = placeObject(obj_img, obj_name_pos, base_img, base_name_pos)
            time_placeObject += time.time() - t
            t = time.time()
        base_img, base_name_pos = filterImages(base_img, base_name_pos)
        time_filterImages += time.time() - t
        t = time.time()
        writeImages(base_img, base_name_pos)
        time_writeImages += time.time() - t
    print("time_makeBase", time_makeBase)
    print("time_objectMake", time_objectMake)
    print("time_objectFilter", time_objectFilter)
    print("time_objectPerspective", time_objectPerspective)
    print("time_placeObject", time_placeObject)
    print("time_filterImages", time_filterImages)
    print("time_writeImages", time_writeImages)


def threadedCreateLabel(base_num, obj_nums, output):
    base_img, base_name_pos = makeBase(base_num)
    for obj_num in obj_nums:
        obj_img, obj_name_pos = objectMake(obj_num)
        obj_img, obj_name_pos = objectFilter(obj_img, obj_name_pos)
        obj_img, obj_name_pos = objectPerspective(obj_img, obj_name_pos)
        base_img, base_name_pos = placeObject(obj_img, obj_name_pos, base_img, base_name_pos)
    base_img, base_name_pos = filterImages(base_img, base_name_pos)
    output[0] = base_img
    output[1] = base_name_pos


def threadedCreateLabels():
    if os.path.exists(DATASET_LOCATION):
        shutil.rmtree(DATASET_LOCATION)
        time.sleep(0.2)
    os.makedirs(DATASET_LOCATION)
    time_makeImage = 0
    time_writeImages = 0
    time_loadBases = 0
    time_loadObjects = 0
    num_threads = DATASET_CREATION_TREADS
    t = time.time()
    print("make object list")
    # filterObjects()
    makeObjectList()
    time_loadBases += time.time() - t
    t = time.time()
    print("make_base_list")
    makeBaseList()
    time_loadObjects += time.time() - t
    t = time.time()
    print("make dataset")
    while len(os.listdir(DATASET_LOCATION)) - 2 < DATASET_NUM_IMAGES:
        num_labels = np.random.choice(list(range(len(DATASET_OBJECT_PLACE_CHANCE))), p=DATASET_OBJECT_PLACE_CHANCE)

        base_num = [baseList.pop(0) for i in range(num_threads)]
        obj_nums = [[objectList.pop(0) for i in range(num_labels)] for i in range(num_threads)]

        threads = [None for i in range(num_threads)]
        results = [[None, None] for i in range(num_threads)]
        for i in range(num_threads):
            threads[i] = threading.Thread(target=threadedCreateLabel, args=(base_num[i], obj_nums[i], results[i]))
            threads[i].start()

        for i in range(num_threads):
            print(".", end="")
            sys.stdout.flush()
            if len(os.listdir(DATASET_LOCATION)) % 100 == 0:
                print(len(os.listdir(DATASET_LOCATION)) - 2)
            threads[i].join()
            base_img, base_name_pos = results[i]
            time_makeImage += time.time() - t
            t = time.time()
            writeImages(base_img, base_name_pos)
            time_writeImages += time.time() - t
            t = time.time()
    print("time_loadBases", time_loadBases)
    print("time_loadObjects", time_loadObjects)
    print("time_makeImage", time_makeImage)
    print("time_writeImages", time_writeImages)


def createDataset():
    print(DATASET_NUM_IMAGES)
    if not os.path.exists("openimages/test") or not os.path.exists("openimages/validation") or REDOWNLOAD_DATASET:
        newDownload()
    if not os.path.exists(DATASET_LOCATION) or len(os.listdir(DATASET_LOCATION)) < DATASET_NUM_IMAGES \
       or REBUILD_DATASET:
        threadedCreateLabels()
