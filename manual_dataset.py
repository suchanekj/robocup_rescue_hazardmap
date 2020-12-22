import os
import random
import string

import cv2
import xml.etree.ElementTree as ET

# file names cannot have " ' "!
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

def rand(a=0., b=1.):
    return np.random.rand() * (b - a) + a

# filter_strength (0, 1): extent of filter application
# colour_filter_strength (0, 1): extent of colour changes
# blue_cut_strength (0, 1): extent of blurring
def objectFilter(srcA, filter_strength, colour_filter_strength, blur_cut_strength):
    rows, cols, colors = srcA.shape
    srcA = np.asarray(srcA, np.uint8)
    randomcol = (255 * rand(), 255 * rand(), 255 * rand(), 0)

    """adds noise to image"""
    if rand() < 0.9 * filter_strength * colour_filter_strength:
        strength = filter_strength * colour_filter_strength
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

    """adds a circle"""
    if rand() < 0.6 * filter_strength * blur_cut_strength:
        strength = filter_strength * blur_cut_strength
        rad = (rows + cols) / 4 * (0.25 + 0.75 * rand())
        thickness = (rows + cols) / 4 * strength * rand()
        cv2.circle(srcA, (int(rows * rand()), int(cols * rand())), int(rad + thickness / 2),
                   randomcol, int(thickness))

    """adds lines"""
    if rand() < 0.6 * filter_strength * blur_cut_strength:
        strength = filter_strength * blur_cut_strength
        for j in range(int(rand() * rand() * 5 * strength) + 1):
            x = [(int(rows * (-0.5 + 2 * rand())), int(cols * (-0.5 + 2 * rand()))) for i in range(2)]
            cv2.line(srcA, x[0], x[1], randomcol, int(1 + rand() * (rows + cols) / 10 * strength))
    #
    """adds text"""
    if rand() < 0.6 * filter_strength * blur_cut_strength:
        strength = filter_strength * blur_cut_strength
        text = np.zeros((rows, cols, 4))
        for j in range(int(1 + 5 * rand() * strength)):
            txt = "".join([random.choice(string.ascii_letters) for i in range(int(15 * rand(0.1, strength)))])
            cv2.putText(text, txt, (int(rows * rand()), int(cols * rand())),
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

    """adds blur"""
    if rand() < 0.25 * filter_strength * blur_cut_strength:
        strength = filter_strength * blur_cut_strength
        srcA = cv2.medianBlur(srcA, 1 + 2 * int(rand(0, strength) ** 3 * rand(0, strength) * (rows + cols) / 40))

    """adds more blur"""
    if rand() < 0.25 * filter_strength * blur_cut_strength:
        strength = filter_strength * blur_cut_strength
        size_1 = 1 + 2 * int(rand(0, strength) ** 3 * rand(0, strength) * (rows + cols) / 200)
        size_2 = 1 + 2 * int(rand(0, strength) ** 3 * rand(0, strength) * (rows + cols) / 200)
        srcA = cv2.blur(srcA, (size_1, size_2))

    """Not sure what this does..."""
    if rand() < 0.3 * filter_strength * blur_cut_strength:
        strength = filter_strength * blur_cut_strength
        srcA = srcA.transpose((2, 0, 1))
        c = 1. - 0.6 * rand(0, strength)
        a = np.linspace(c + rand(0, strength) * (1 - c), c + rand(0, strength) * (1 - c), rows)[:, None]
        b = np.linspace(c + rand(0, strength) * (1 - c), c + rand(0, strength) * (1 - c), cols)[None, :]
        gradient = np.multiply(a, b)
        srcA[0:3] = 255 - np.multiply(255 - srcA[0:3], gradient)
        srcA = srcA.transpose((1, 2, 0))

    """Not sure what this does..."""
    if rand() < 0.5 * filter_strength * blur_cut_strength:
        strength = filter_strength * blur_cut_strength
        srcA = srcA.transpose((2, 0, 1))
        c = 1. - 0.6 * rand(0, strength)
        a = np.linspace(c + rand(0, strength) * (1 - c), c + rand(0, strength) * (1 - c), rows)[:, None]
        b = np.linspace(c + rand(0, strength) * (1 - c), c + rand(0, strength) * (1 - c), cols)[None, :]
        gradient = np.multiply(a, b)
        srcA[0:3] = np.multiply(srcA[0:3], gradient)
        srcA = srcA.transpose((1, 2, 0))

    """changes the colours!"""
    if rand() < 0.95 * filter_strength * colour_filter_strength:
        strength = filter_strength * blur_cut_strength
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

    return srcA

def objectPerspective(srcA, pointsList, filter_strength, perspective_strength, rotation_strength, distortion_strength, erosion_strength):
    """
    filter_strength (0, 1): to what extent should the image be modified
    perspective_strength (-2, 2): extent of perspective change
    rotation_strength (0, 1): extent of rotation
    distortion_strength (0, 1): extent of distortion
    erosion_strength (0, 15): extent of erosion
    """
    rows, cols, colors = srcA.shape

    corners = []
    img = srcA
    imgOriginal = srcA

    # Warp
    # Perspective
    pts1 = np.float32([
        [0, 0],
        [rows, 0],
        [0, cols],
        [rows, cols]
    ])

    def random_around_P(v):
        dist_y = rows//4 * filter_strength * rand(-1, 1)
        dist_x = cols//4 * filter_strength * rand(-1, 1)
        return np.array([v[0] + dist_y, v[1] + dist_x])

    pts2 = np.float32([random_around_P(x) for x in pts1])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    # img = cv2.warpPerspective(img, M, (rows, cols))
    img = cv2.warpPerspective(img, M, (cols, rows))
    for a in range(len(pointsList)):
        pointsList[a] = np.float32(cv2.perspectiveTransform(pointsList[a].reshape(1, -1, 2), M).reshape(-1, 2))

    # Rotation 1 / 2
    M = cv2.getRotationMatrix2D((cols, rows), (1 - 2 * rand()) * 180 * rotation_strength / 2, 1)

    img = cv2.warpAffine(img, M, (cols, rows))
    for a in range(len(pointsList)):
        pointsList[a] = np.int32(cv2.transform(pointsList[a].reshape(1, -1, 2), M).reshape(-1, 2))

    # Camera distortion
    """changes the position significantly, so can't be used"""
    # DIM = img.shape[:2]
    # K = np.array([[img.shape[0] * rand(0.4, 0.6), 0.0, img.shape[0] * rand(0.4, 0.6)],
    #               [0.0, img.shape[0] * rand(0.4, 0.6), img.shape[0] * rand(0.4, 0.6)],
    #               [0.0, 0.0, 1.0]])
    # if rand(-1, 1) > 0:
    #     center = 3 * distortion_strength * filter_strength
    #     scale = 4 * distortion_strength * filter_strength
    # else:
    #     center = -0.2 * distortion_strength * filter_strength
    #     scale = 0.5 * distortion_strength * filter_strength
    # D = np.array([[0.707107 + center + scale * rand(-1, 1)], [center + scale * rand(-1, 1)],
    #               [0.707107 + center + scale * rand(-1, 1)], [center + scale * rand(-1, 1)]])
    #
    # map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    # img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Rotation 2 / 2
    M = cv2.getRotationMatrix2D((cols, rows), (1 - 2 * rand()) * 180 * rotation_strength / 2, 1)

    img = cv2.warpAffine(img, M, (cols, rows))
    for a in range(len(pointsList)):
        pointsList[a] = np.int32(cv2.transform(pointsList[a].reshape(1, -1, 2), M).reshape(-1, 2))

    # Erode alpha
    img[:, :, 3] = np.round(img[:, :, 3] / 255).astype(img.dtype) * 255
    erosion_size = int(rand(1, erosion_strength))
    erosion_type = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(erosion_type, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                        (erosion_size, erosion_size))
    img[:, :, 3] = cv2.erode(img[:, :, 3], element)

    # for points in pointsList:
    #     print(points)
    #     points = np.array(points, np.int32)
    #     temp = points[2]
    #     points[2] = points[1]
    #     points[1] = temp
    #     # points[1], points[2] = points[2], points[1]
    #     points = points.reshape((-1,1,2))
    #     # points = np.int32(np.array(points))
    #     img = cv2.polylines(img, [points],True, (255,0,0), 10)
    #     # cv2.fillPoly(img, pts=[points], color=(255, 0, 0))

    for points in pointsList:
        for i, point in enumerate(points):
            if i == 0:
                point[0] = max(point[0], 0)
                point[1] = max(point[1], 0)
            if i == 1:
                point[0] = min(point[0], cols)
                point[1] = max(point[1], 0)
            if i == 2:
                point[0] = max(point[0], 0)
                point[1] = min(point[1], rows)
            if i == 3:
                point[0] = min(point[0], cols)
                point[1] = min(point[1], rows)
        corners.append([points[0],
                        points[3]])

    ret, imgThreshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    imgThreshold = np.int32(np.amax(imgThreshold, axis=2))
    mask = 255 - imgThreshold
    masks = [mask for _ in range(3)]
    masks.append(np.zeros((rows, cols)))
    masks = np.stack(masks, axis=2)

    # blur background image
    factor = 100
    kernel = np.ones((factor, factor), np.float32) / (factor * factor)
    imgOriginal = cv2.filter2D(imgOriginal, -1, kernel)

    img2 = np.where(masks>0, imgOriginal, 0)
    img = img2 + img

    # Debugging - draws rectangles
    # for i in corners:
    #     cv2.rectangle(img, tuple(i[0]), tuple(i[1]), (255,0,0), 10)

    return img, corners[:-1]

def imageMaker(makeImageCount, imageSize, filterStrength, xmlPath, labelNamesPath, imageInputPath, imageOutputPath, labelsPath, outputLabelNamesPath):
    if not os.path.exists(imageOutputPath[:-1]):
        os.makedirs(imageOutputPath[:-1])

    rows2, cols2 = imageSize
    root = ET.parse(xmlPath).getroot()
    # ET skips over other tags. images is 2nd item
    imageXMLList = root[2]

    labelList = []
    labelValueDict = {}
    fullImagePathList = []
    partialImagePathList = []

    # Reads labelNames.txt and creates a dictionary of values
    with open(labelNamesPath, 'r') as myfile:
        with open(outputLabelNamesPath, 'a') as myfile2:
            currLine = myfile.readline()

            # each line is like 1 l/n/fire_extinguisher_sign!
            while currLine != "":
                myfile2.write(currLine)
                currLine = currLine.split()
                labelValueDict[currLine[1]] = int(currLine[0])
                currLine = myfile.readline()

    #gathers list of image paths
    for imageXML in imageXMLList:
        partialImagePath = imageXML.attrib['file']
        fullImagePath = imageInputPath + partialImagePath
        partialImagePathList.append(partialImagePath)
        fullImagePathList.append(fullImagePath)

    imageCount = len(fullImagePathList)

    #randomise order of images
    indexList = [i for i in range(imageCount)]

    #for every index, if we loop over the entire set of images, shuffle the images
    for j in range(makeImageCount):
        r = j % imageCount
        # if r == 0:
        #     random.shuffle(indexList)
        i = indexList[r]

        image = cv2.imread(fullImagePathList[i])
        # partialImagePath = partialImagePathList[i]
        partialImagePath = j

        if image is not None:
            fullImageOutputPath = os.getcwd() + "/" + imageOutputPath + str(j) + ".png"
            imgRows, imgCols, _ = image.shape

            # adding alpha
            b1, g1, r1 = cv2.split(image)
            a1 = np.zeros(b1.shape, dtype=b1.dtype)
            image = cv2.merge((b1, g1, r1, a1))

            label = [fullImageOutputPath]

            boxList = []
            labelValueList = []
            # for every box with a detection, add a tuple with
            # 4 coordinates and the label to image row
            # bottom left y, bottom left x, height, width
            for boxes in imageXMLList[i]:
                labelName = boxes[0].text

                if labelName in labelValueDict:
                    labelValue = labelValueDict[labelName]
                    boxCoords = boxes.attrib
                    coordTuple = ()

                    for coord in boxCoords:
                        coordTuple += (int(boxCoords[coord]),)

                    py, px, width, height = coordTuple

                    boxList.append(np.float32([
                        [px / imgCols * cols2, py / imgRows * rows2],
                        [(px + width) / imgCols * cols2, py / imgRows * rows2],
                        [px / imgCols * cols2, (py + height) / imgRows * rows2],
                        [(px + width) / imgCols * cols2, (py + height) / imgRows * rows2]
                    ]))

                    labelValueList.append(labelValue)
                    coordTuple += (labelValue,)

            # adds the corners of the image so we can use the corners to make a mask
            boxList.append(np.float32([
                [0, 0],
                [cols2, 0],
                [0, rows2],
                [cols2, rows2]
            ]))

            # adds filters to images
            image = cv2.resize(image, (cols2, rows2))
            image = objectFilter(image, filterStrength, filterStrength/10, filterStrength/10)
            # image, cornerPairList = objectPerspective(image, boxList, filterStrength, filterStrength, filterStrength, 0, 15*filterStrength)
            image, cornerPairList = objectPerspective(image, boxList, filterStrength, filterStrength, 0, filterStrength, 15*filterStrength)
            cv2.imwrite(fullImageOutputPath, image)
            image = cv2.imread(fullImageOutputPath)
            #save image to location
            cv2.imwrite(fullImageOutputPath[:-3] + "png", image)

            #append corners to each image row listing

            for cornerPairIdx in range(len(cornerPairList)):
                coordList = []
                cornerPair = cornerPairList[cornerPairIdx]
                labelValue = labelValueList[cornerPairIdx]

                for corner in cornerPair:
                    for xyvals in corner:
                        coordList.append(xyvals)
                coordList.append(labelValue)
                label.append(coordList)
            labelList.append(label)

    with open(labelsPath, 'w') as myfile:
        for label in labelList:
            #each label is:
            #['data/images/yoko.png', [194, 110, 291, 189, 32], [169, 118, 300, 417, 31]]
            content = ""
            for argIdx in range(len(label)):
                if argIdx == 0:
                    content += str(label[argIdx]) + " "
                else:
                    content += str(label[argIdx])[1:-1] + " "
            content += "\n"
            myfile.write(content)



if __name__ == "__main__":
    filterStrengthTuple = (0.1, 0.4, 1., 1., 0.9)
    imageSizeTuple = ((96, 128), (192, 256), (288, 384), (480, 640), (480, 640))

    imageCount = 10

    xmlPath = "manualy_labeled/labels_1.xml"
    labelNamesPath = "labelNames.txt"
    imageInputPath = "manualy_labeled/labeled_1/"

    sharedPath = "datasets2/dataset_open_"

    imageOutputPathList = [sharedPath + str(i) + "/" for i in range(5)]
    labelsPathList = [sharedPath + str(i) + "/labels.txt" for i in range(5)]
    labelNamesPathList = [sharedPath + str(i) + "/labelNames.txt" for i in range(5)]

    for i in range(5):
        filterStrength = filterStrengthTuple[i]
        imageSize = imageSizeTuple[i]

        imageOutputPath = imageOutputPathList[i]
        labelsPath = labelsPathList[i]
        outputLabelNamesPath = labelNamesPathList[i]

        imageMaker(imageCount, imageSize, filterStrength, xmlPath, labelNamesPath, imageInputPath, imageOutputPath, labelsPath, outputLabelNamesPath)