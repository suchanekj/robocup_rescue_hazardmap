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
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

from config import *


labelList = []
labels = []
labelNames = []
labelIDs = {}
baseList = []
bases = []


def newDownload(top_layer=True):
    try:
        downloaded = open("downloaded.txt", "r")
        downloaded_list = downloaded.readlines()
        downloaded.close()
    except:
        downloaded_list = []

    for i in range(len(downloaded_list)):
        downloaded_list[i] = downloaded_list[i][0:-1]
    downloaded_list = set(downloaded_list)
    # print(downloaded_list)
    downloaded_list = downloaded_list.intersection(DATASET_DOWNLOAD_KEYWORDS)
    validFiles = []
    for filename in os.listdir("downloads"):
        # print(filename[0:-6])
        if filename[0:-7] in downloaded_list:
            validFiles.append(filename)
    toDelete = set(os.listdir("downloads")) - set(validFiles)
    # print(toDelete)
    for filename in toDelete:
        os.remove("downloads/" + filename)
    dataset_keywords = list(set(DATASET_DOWNLOAD_KEYWORDS) - downloaded_list)

    response = google_images_download.googleimagesdownload()
    to_download = int(DATASET_NUM_IMAGES * 2) - len(os.listdir("downloads"))
    num_per_key = int(to_download / max(1, len(dataset_keywords)) / DATASET_DOWNLOAD_TIME_MUL)
    num_extra_img = to_download - num_per_key * len(dataset_keywords)
    if not os.path.isfile("downloaded.txt"):
        downloaded = open("downloaded.txt", "w+")
        downloaded.close()

    for j, i in enumerate(dataset_keywords):
        for time_shift in range(DATASET_DOWNLOAD_TIME_MUL):
            try:
                start_date = (datetime.date.today() -
                              datetime.timedelta(days=3650/DATASET_DOWNLOAD_TIME_MUL *
                                                      (time_shift + 1))).strftime("%d/%m/%Y")
                end_date = (datetime.date.today() -
                            datetime.timedelta(days=3650/DATASET_DOWNLOAD_TIME_MUL *
                                                    time_shift)).strftime("%d/%m/%Y")
                response.download({"keywords": i, "limit": min(num_per_key + (1 if j < num_extra_img else 0), 100),
                                   "exact_size": "640,480", "image_directory": ".",
                                   "time_range":'{"time_min":"'+start_date+'","time_max":"'+end_date+'"}'})
            except Exception as e:
                print(e)
            else:
                if min(num_per_key + (1 if j < num_extra_img else 0), 100) == 100:
                    print("keyword finished")
                    downloaded = open("downloaded.txt", "a+")
                    downloaded.write(i + "\n")
                    downloaded.close()
        index = 0
        for filename in os.listdir("downloads"):
            try:
                if filename[0].isdigit():
                    os.rename("downloads/" + filename, "downloads/" + i + str(index).zfill(3) + ".jpg")
                    index += 1
            except Exception as e:
                print(e)
    if top_layer:
        newDownload(top_layer=False)


def downloadsFilter():
    for filename in os.listdir("downloads"):
        if filename.endswith('.jpg'):
            try:
                src = cv2.imread("downloads/" + filename, cv2.IMREAD_COLOR)
                rows, cols, colors = src.shape
                if rows != 480 or cols != 640 or colors != 3:
                    raise ValueError("lol its d rong sheip")
                img = Image.open('downloads/' + filename)  # open the image file
                img.verify()  # verify that it is, in fact an image
            except Exception as e:
                print(e)
                os.remove('downloads/' + filename)
        else:
            os.remove('downloads/' + filename)


def labelsStraiten():
    FListA = os.listdir("labelsAngled")
    FListS = os.listdir("labelsStraight")
    for i in list(set(FListA) - set(FListS)):
        img = cv2.imread("labelsAngled/" + i, cv2.IMREAD_UNCHANGED)
        rows, cols, colors = img.shape

        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        crop_img = dst[int(cols*(1/2-1/np.sqrt(8))):int(cols*(1/2+1/np.sqrt(8))),
                       int(rows*(1/2-1/np.sqrt(8))):int(rows*(1/2+1/np.sqrt(8)))]
        cv2.imwrite("labelsStraight/" + i, crop_img)


def makeLabelList():
    global labels
    global labelList
    global labelNames
    global labelIDs
    FList = os.listdir("temp/straight/")
    f = open(DATASET_LOCATION + "/labelNames.txt", "w+")
    for i in FList:
        img = cv2.imread("temp/straight/" + i, cv2.IMREAD_UNCHANGED)
        labels.append(img)
        labelNames.append(i[0:-4])
        labelIDs[i[0:-4]] = len(labelNames) - 1
        f.write(str(len(labelNames) - 1) + " " + i[0:-4] + "\n")
    for i in range(int(DATASET_MAX_LABELS_PER_IMG * DATASET_NUM_IMAGES / len(labels))):
        x = list(range(len(labels)))
        random.shuffle(x)
        labelList.extend(x)
    f.close()


def labelMake():
    out = [None,None]
    if random.random() < DATASET_FOURS_PART:
        x = []
        for j in range(4):
            a = copy.deepcopy(labels[labelList[j]])
            rows, cols, colors = a.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2),
                                        90 * int(random.random() * 4) - DATASET_FOUR_MEMBER_WIGGLE
                                        + int(random.random() * (1 + DATASET_FOUR_MEMBER_WIGGLE * 2)), 1)
            a = cv2.warpAffine(a, M, (cols, rows))
            M = np.float32([[1, 0, - DATASET_FOUR_MEMBER_WIGGLE * 2 +
                             int(random.random() * (4 * DATASET_FOUR_MEMBER_WIGGLE + 1))],
                            [0, 1, - DATASET_FOUR_MEMBER_WIGGLE * 2 +
                             int(random.random() * (4 * DATASET_FOUR_MEMBER_WIGGLE + 1))]])
            a = cv2.warpAffine(a, M, (cols, rows))
            x.append(a)

        cat0 = np.concatenate((x[0], x[1]), axis=0)
        cat1 = np.concatenate((x[2], x[3]), axis=0)
        four = np.concatenate((cat0, cat1), axis=1)
        name = labelNames[labelList[0]] + ";" + labelNames[labelList[1]] + ";" + \
               labelNames[labelList[2]] + ";" + labelNames[labelList[3]]

        for a in range(4):
            labelList.pop(0)

        out[0] = four
        out[1] = name
    else:
        out[0] = copy.deepcopy(labels[labelList[0]])
        out[1] = labelNames[labelList.pop(0)]

    return out


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


def labelFilter(srcA):
    rows, cols, colors = srcA.shape

    if random.random() < 0.9 * DATASET_FILTERING_MODIFIER:
        noise = np.zeros((rows, cols, 4))
        m = np.array([255 * random.random(), 255 * random.random(), 255 * random.random(), 10 * random.random()])
        sigma = np.array([80 * random.random(), 80 * random.random(), 80 * random.random(), 1 * random.random()])
        cv2.randn(noise, m, sigma)
        noise = (noise > 0) * noise - (noise > 255) * (noise - 255)
        noise = noise.transpose((2, 0, 1))
        srcA = srcA.transpose((2, 0, 1))
        srcA[0:3] = np.multiply(srcA[0:3], (1 - noise[3] / 255)) + np.multiply(noise[0:3], noise[3] / 255)
        srcA[3] = np.multiply(srcA[3], (1 - noise[3] / 255)) + noise[3]
        srcA = srcA.transpose((1, 2, 0))

    if random.random() < 0.5 * DATASET_FILTERING_MODIFIER:
        rad = 100 + 300 * random.random()
        thickness = 300
        cv2.circle(srcA, (int(rows * random.random()), int(cols *random.random())), int(rad + thickness/2),
                   (0, 0, 0, 0), thickness)

    if random.random() < 0.4 * DATASET_FILTERING_MODIFIER:
        for j in range(int(random.random() * random.random() * 5) + 1):
            x = [(int(rows * (-0.5 + 2 * random.random())), int(cols * (-0.5 + 2 * random.random()))) for i in range(2)]
            cv2.line(srcA, x[0], x[1], (0, 0, 0, 0), int(10 + random.random() * 65))

    if random.random() < 0.5 * DATASET_FILTERING_MODIFIER:
        text = np.zeros((rows, cols, 4))
        for j in range(int(1 + 5 * random.random())):
            txt = "".join([random.choice(string.ascii_letters) for i in range(15)])
            cv2.putText(text, txt, (int(rows * random.random()), int(cols *random.random())),
                        cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1 + random.random() * 5,
                        (int(random.random() * 255), int(random.random() * 255), int(random.random() * 255), 255),
                        int(5 + random.random() * 10))
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 360 * random.random(), 1)
        text = cv2.warpAffine(text, M, (cols, rows))
        text = text.transpose((2, 0, 1))
        srcA = srcA.transpose((2, 0, 1))
        srcA[0:3] = np.multiply(srcA[0:3], (1 - text[3] / 255)) + np.multiply(text[0:3], text[3] / 255)
        srcA[3] = np.multiply(srcA[3], (1 - text[3] / 255)) + text[3]
        srcA = srcA.transpose((1, 2, 0))

    if random.random() < 0.2 * DATASET_FILTERING_MODIFIER:
        srcA = cv2.medianBlur(srcA, 3 + 2* int(random.random()**3 * 20))

    if random.random() < 0.2 * DATASET_FILTERING_MODIFIER:
        srcA = cv2.blur(srcA, (3 + 2 * int(random.random()**3 * 4), 3 + 2 * int(random.random()**3 * 4)))

    if random.random() < 0.5 * DATASET_FILTERING_MODIFIER:
        srcA = srcA.transpose((2, 0, 1))
        c = 0.5
        a = np.linspace(c + random.random() * (1 - c), c + random.random() * (1 - c), rows)[:, None]
        b = np.linspace(c + random.random() * (1 - c), c + random.random() * (1 - c), cols)[None, :]
        gradient = np.multiply(a, b)
        srcA[0:3] = 255 - np.multiply(255 - srcA[0:3], gradient)
        srcA = srcA.transpose((1, 2, 0))

    if random.random() < 0.7 * DATASET_FILTERING_MODIFIER:
        srcA = srcA.transpose((2, 0, 1))
        c = 0.3
        a = np.linspace(c + random.random() * (1 - c), c + random.random() * (1 - c), rows)[:, None]
        b = np.linspace(c + random.random() * (1 - c), c + random.random() * (1 - c), cols)[None, :]
        gradient = np.multiply(a, b)
        srcA[0:3] = np.multiply(srcA[0:3], gradient)
        srcA = srcA.transpose((1, 2, 0))

    if random.random() < 0.8 * DATASET_FILTERING_MODIFIER:
        hue = 0.1
        sat = 1.5
        val = 1.5
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        src = srcA[:,:,0:3]
        x = rgb_to_hsv(src / 255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        src = hsv_to_rgb(x) * 255
        srcA[:,:,0:3] = src

    return srcA


def labelPerspective(input):
    src = input[0]
    names = input[1].split(';')
    rows, cols, colors = src.shape
    if len(names) == 4:
        points = [np.float32([[int(rows / 2), int(rows / 2)], [rows, int(rows / 2)],
                             [int(rows / 2), rows], [rows, rows]]),
                  np.float32([[rows, int(rows / 2)], [int(rows / 2) + rows, int(rows / 2)],
                             [rows, rows], [int(rows / 2) + rows, rows]]),
                  np.float32([[int(rows / 2), rows], [rows, rows],
                             [int(rows / 2), int(rows / 2) + rows], [rows, int(rows / 2) + rows]]),
                  np.float32([[rows, rows], [int(rows / 2) + rows, rows],
                             [rows, int(rows / 2) + rows], [int(rows / 2) + rows, int(rows / 2) + rows]])]
    else:
        points = [np.float32([[int(rows / 2), int(rows / 2)],
                             [int(rows / 2) + rows, int(rows / 2)],
                             [int(rows / 2), int(rows / 2) + rows],
                             [int(rows / 2) + rows, int(rows / 2) + rows]])]

    corners = []
    img = np.zeros((rows * 2, cols * 2, 4))
    img[int(rows / 2):int(rows / 2) + rows,int(rows / 2):int(rows / 2) + rows] = src

    a = 1/2 - 0.3 * DATASET_FILTERING_MODIFIER
    pts1 = np.float32([[int(rows / 2), int(rows / 2)], [int(rows / 2) + rows, int(rows / 2)],
                       [int(rows / 2), int(rows / 2) + rows], [int(rows / 2) + rows, int(rows / 2) + rows]])
    pts2 = np.float32([[int(rows * (a + random.random() * (1 - 2 * a))),
                        int(rows * (a + random.random() * (1 - 2 * a)))],
                       [int(rows * (a + random.random() * (1 - 2 * a))) + rows,
                        int(rows * (a + random.random() * (1 - 2 * a)))],
                       [int(rows * (a + random.random() * (1 - 2 * a))),
                        int(rows * (a + random.random() * (1 - 2 * a))) + rows],
                       [int(rows * (a + random.random() * (1 - 2 * a))) + rows,
                        int(rows * (a + random.random() * (1 - 2 * a))) + rows]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img,M,(rows * 2, cols * 2))
    for a in range(len(points)):
        points[a] = np.float32( cv2.perspectiveTransform(points[a].reshape(1, -1, 2), M).reshape(-1, 2))

    M = cv2.getRotationMatrix2D((cols, rows), 360 * random.random(), 1)
    img = cv2.warpAffine(img, M, (rows * 2, cols * 2))
    for a in range(len(points)):
        points[a] = np.int32( cv2.transform(points[a].reshape(1, -1, 2), M).reshape(-1, 2))
        corners.append([(np.min(points[a].transpose()[0]), np.min(points[a].transpose()[1])),
                        (np.max(points[a].transpose()[0]), np.max(points[a].transpose()[1]))])
        # cv2.rectangle(img, corners[-1][0], corners[-1][1], (255, 0, 255, 255), 20)

    namesPositions = []
    for i in range(len(names)):
        namesPositions.append([names[i]])
        namesPositions[-1].append(corners[i][0][0])
        namesPositions[-1].append(corners[i][0][1])
        namesPositions[-1].append(corners[i][1][0])
        namesPositions[-1].append(corners[i][1][1])

    # print(namesPositions)
    return [img, namesPositions]


def placeLabel(input):
    label = input[0]
    namesPositions = input[1]

    base = cv2.imread("downloads/" + bases[baseList.pop(0)], cv2.IMREAD_COLOR)
    size = 0.02 + 0.98 * (1 - DATASET_FILTERING_MODIFIER) +\
           0.98 * random.random() * random.random() * DATASET_FILTERING_MODIFIER
    label = cv2.resize(label, None, fx=size, fy=size, interpolation=cv2.INTER_AREA)
    lrows, lcols, lcolors = label.shape
    brows, bcols, bcolors = base.shape

    position = (int(random.random() * (brows - lrows * (1 - DATASET_FILTERING_MODIFIER))
                    - lrows / 2 * DATASET_FILTERING_MODIFIER),
                int(random.random() * (bcols - lcols * (1 - DATASET_FILTERING_MODIFIER))
                    - lcols / 2 * DATASET_FILTERING_MODIFIER))
    label = label.transpose((2, 0, 1))
    base = base.transpose((2, 0, 1))

    base[0:3, max(0, position[0]):min(brows, position[0] + lcols),
                         max(0, position[1]):min(bcols, position[1] + lrows)] =\
        np.multiply(base[0:3, max(0, position[0]):min(brows, position[0] + lcols),
                         max(0, position[1]):min(bcols, position[1] + lrows)],
                    (1 - label[3, max(0, -position[0]):min(lrows, brows - position[0]),
                               max(0, -position[1]):min(lcols, bcols - position[1])] / 255))\
        + np.multiply(label[0:3, max(0, -position[0]):min(lrows, brows - position[0]),
                               max(0, -position[1]):min(lcols, bcols - position[1])],
                      label[3, max(0, -position[0]):min(lrows, brows - position[0]),
                               max(0, -position[1]):min(lcols, bcols - position[1])] / 255)
    base = base.transpose((1, 2, 0))

    for a in range(len(namesPositions)):
        for b in range(2):
            namesPositions[a][1 + 2 * b] = max(min(round(int(namesPositions[a][1 + 2 * b])*size) + position[1], bcols), 0)
            namesPositions[a][2 + 2 * b] = max(min(round(int(namesPositions[a][2 + 2 * b])*size) + position[0], brows), 0)
        # cv2.rectangle(base, tuple(x[a][0]), tuple(x[a][1]), (255, 0, 255, 255), 20)
    return [base, namesPositions]


def filterImages(input):
    base = input[0]
    namesPositions = input[1]

    rows, cols = base.shape[0:2]

    if random.random() < 0.4 * DATASET_FILTERING_MODIFIER:
        x = np.sin(2 * np.pi * random.random())
        y = np.sqrt(1 - x**2)
        src = base
        reps = int(random.random()**2 * 20)
        base = base / (2 * reps + 1)

        for i in range(reps):
            M1 = np.float32([[1, 0, round(x * (i + 1) / 2)], [0, 1, round(y * (i + 1) / 2)]])
            M2 = np.float32([[1, 0, -round(x * (i + 1) / 2)], [0, 1, -round(y * (i + 1) / 2)]])
            base += (cv2.warpAffine(src, M1, (cols, rows)) / (2 * reps + 1)
                     + cv2.warpAffine(src, M2, (cols, rows)) / (2 * reps + 1))

    if random.random() < 0.2 * DATASET_FILTERING_MODIFIER:
        base = cv2.medianBlur(base, 1 + 2 * int(random.random()**3 * 10))

    if random.random() < 0.2 * DATASET_FILTERING_MODIFIER:
        base = cv2.blur(base, (1 + 2 * int(random.random()**3 * 3), 1 + 2 * int(random.random()**3 * 3)))

    if random.random() < 0.5 * DATASET_FILTERING_MODIFIER:
        hue = 0.1
        sat = 1.5
        val = 1.5
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

    return [base, namesPositions]


def writeImages(input):
    base = input[0]
    namesPositions = input[1]
    f = open(DATASET_LOCATION + "/labels.txt", "a+")

    index = len(os.listdir(DATASET_LOCATION)) - 2
    f.write(os.getcwd() + "\\" + DATASET_LOCATION + "\\" + str(index).zfill(7) + ".png ")
    for a in range(len(namesPositions)):
        if (namesPositions[a][1] - namesPositions[a][3]) * (namesPositions[a][2] - namesPositions[a][3]) == 0:
            continue
        f.write(','.join(map(str, namesPositions[a][1:])))
        f.write(',')
        f.write(str(labelIDs[namesPositions[a][0]]))
        if a != len(namesPositions) - 1:
            f.write(' ')
    f.write('\n')
    cv2.imwrite(DATASET_LOCATION + "\\" + str(index).zfill(7) + ".png", base)
    f.close()


def createLabels():
    if os.path.exists(DATASET_LOCATION):
        shutil.rmtree(DATASET_LOCATION)
        time.sleep(0.2)
    os.mkdir(DATASET_LOCATION)
    labelsStraiten()
    makeLabelList()
    while len(os.listdir(DATASET_LOCATION)) - 3 < DATASET_NUM_IMAGES:
        pair = labelMake()
        pair[0] = labelFilter(pair[0])
        pair = labelPerspective(pair)
        pair = placeLabel(pair)
        try:
            pair = filterImages(pair)
        except Exception as e:
            print(e)
            pass
        else:
            writeImages(pair)


def createDataset():
    global DATASET_NUM_IMAGES
    global baseList
    global bases
    print(DATASET_NUM_IMAGES)
    if not os.path.exists("downloads") or len(os.listdir("downloads")) < DATASET_NUM_IMAGES:
        newDownload()
        downloadsFilter()
    DATASET_NUM_IMAGES = DATASET_MAX_NUM_IMAGES
    bases = os.listdir("downloads")
    # while len(baseList) < DATASET_NUM_IMAGES + 100:
    for i in range(int(np.ceil(DATASET_NUM_IMAGES / len(bases))) + 2):
        baseList.extend(list(range(len(bases))))
        print(i)
    random.shuffle(baseList)
    # print(baseList)
    # print(len(bases))
    if not os.path.exists(DATASET_LOCATION) or len(os.listdir(DATASET_LOCATION)) < DATASET_NUM_IMAGES \
       or REBUILD_DATASET:
        createLabels()
