from pandas import np
from PIL import Image
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


class ConfusionMatrix:
    def __init__(self, n: int):
        self.n = n
        self.mat_count = [[0] * n for _ in range(n)]
        self.mat_frac = [[0] * n for _ in range(n)]
        self.actual_class_counts = [0] * n
        self.predicted_class_counts = [0] * n

    def add_predicted(self, predicted_class):
        assert predicted_class < self.n
        self.predicted_class_counts[predicted_class] += 1

    def add_actual(self, actual_class):
        assert actual_class < self.n
        self.actual_class_counts[actual_class] += 1

    def add_actual_predicted(self, actual_class: int, predicted_class: int):
        assert predicted_class < self.n
        assert actual_class < self.n
        self.mat_count[actual_class][predicted_class] += 1

    def get_count_matrix(self) -> list:
        return self.mat_count

    def get_fraction_matrix(self) -> list:
        for i, total in enumerate(self.actual_class_counts):
            row = self.mat_count[i]
            for j, v in enumerate(row):
                if total == 0:  # if there are no instances in the first place, set them to None
                    self.mat_frac[i][j] = None
                else:
                    self.mat_frac[i][j] = v / total
        return self.mat_frac

    def get_num_predictions(self) -> int:
        return sum(self.predicted_class_counts)

    def get_num_actual(self) -> int:
        return sum(self.actual_class_counts)

    def get_num_predictions_correct(self) -> int:
        ctr = 0
        for i in range(self.n):
            ctr += self.mat_count[i][i]
        return ctr

    def get_precision(self) -> float:
        return self.get_num_predictions_correct() / self.get_num_predictions()

    def get_recall(self) -> float:
        return self.get_num_predictions_correct() / self.get_num_actual()

    def get_iou(self) -> float:
        return self.get_num_predictions_correct() / (
                self.get_num_predictions() + self.get_num_actual() - self.get_num_predictions_correct())

    def process(self, actual_list, predicted_list):
        """
        @param actual_list
            list of (coordinates + class)
            e.g. [[316, 128, 490, 335, 12], [58, 177, 294, 465, 18], [600, 114, 640, 352, 19]]

        @param predicted_list
            list of (coordinates + class + thresholds)
            e.g. [[252, 174, 547, 432, 33, 0.24268064], [227, 54, 514, 292, 33, 0.27982953]]
        """
        """
        Confusion matrix pseudocode:

        for each actual box:
            for each predicted box:
                if the predicted box overlaps more than threshold with an actual box:
                    assign the predicted box to the actual box
                    remove the predicted box from the list of predicted boxes

        now, for each actual box, we have a list of candidate boxes. These candidate boxes 
        might have been predicted correctly or wrongly, we just know that they overlap

        for each actual box:
            for each predicted box related to the actual box:
                add 1 to confusion_matrix[actual_class][predicted_class]

        Problems: 
            Note that if we have clustering, then we may assign the predicted boxes wrongly to the actual boxes.
            I can't think of a way around this, so I'm leaving it as it is right now.
        """

        # bookkeeping: counts the number of actual labels and predicted labels
        for actual_tuple in actual_list:
            actual_label = actual_tuple[4]
            self.add_actual(actual_label)

        for predicted_tuple in predicted_list:
            predicted_label = predicted_tuple[4]
            self.add_predicted(predicted_label)

        memo = [[] for _ in range(len(actual_list))]

        for i, actual_tuple in enumerate(actual_list):
            actual_box = actual_tuple[:4]
            idx_to_add = []
            for j, predicted_tuple in enumerate(predicted_list):
                predicted_box = predicted_tuple[:4]
                if intersection_over_union(actual_box, predicted_box) > 0.5:
                    idx_to_add.append(j)

            idx_to_add.reverse()
            for idx in idx_to_add:
                predicted_tuple = predicted_list[idx]
                memo[i].append(predicted_tuple)
                del predicted_list[idx]

        for i, actual_tuple in enumerate(actual_list):
            actual_label = actual_tuple[4]
            related_tuples = memo[i]
            for j, predicted_tuple in enumerate(related_tuples):
                predicted_label = predicted_tuple[4]
                self.add_actual_predicted(actual_label, predicted_label)


def area_box(box):
    """
    @param box
        box coordinates, note that (0,0) is the top left corner

    @return box's area
    """
    left, top, right, bottom = box
    return (right - left) * (bottom - top)


def area_intersection(box1, box2):
    """
    @param box1
        box coordinates
    @param box2
        box coordinates

    @return area
        returns area of intersection, which is 0 if they don't intersect
    """
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2

    left = max(left1, left2)
    top = max(top1, top2)
    bottom = min(bottom1, bottom2)
    right = min(right1, right2)
    if right > left and bottom > top: return (right - left) * (bottom - top)
    return 0


def intersection_over_union(box1, box2):
    intersection_area = area_intersection(box1, box2)
    return intersection_area / (
            area_box(box1) + area_box(box2) - intersection_area)


def evaluate(test_lines, yolo, log_dir):
    """
    This function is called once every training cycle.

    @param test_lines
        list of strings in the following format
        ['/data/programming/robocup_rescue_hazardmap/datasets/validation_dataset_large/0.png 223,201,249,233,23\n',
         '/data/programming/robocup_rescue_hazardmap/datasets/validation_dataset_large/1.png 297,226,595,469,32 463,102,541,184,19\n']

    @param yolo
        yolo model

    @param log_dir: string
        has the following format:
        logs/018/

    @side-effect
        writes data to txt file in log_dir
    """
    CLASS_COUNT = 35
    confusion_matrix = ConfusionMatrix(CLASS_COUNT)

    for line_num, line in enumerate(test_lines):
        if line[-1] == '\n':
            line = line[:-1]
        lines = line.split(" ")
        path = lines[0]

        correct_boxes = [[int(x) for x in txt.split(',')] for txt in lines[1:]]

        r_image = Image.open(path)

        predicted_boxes = yolo.detect_boxes(r_image, True)

        confusion_matrix.process(correct_boxes, predicted_boxes)

    # for row in confusion_matrix.get_fraction_matrix():
    #     print(row)
    df_cm = pd.DataFrame(confusion_matrix.get_count_matrix(), index=[i for i in range(CLASS_COUNT)],
                         columns=[i for i in range(CLASS_COUNT)])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(log_dir + "confusion_matrix.png")

    with open(log_dir + "validation.txt", "a") as f:
        f.write(f"Correct Predictions Made By Model: {confusion_matrix.get_num_predictions_correct()}\n")
        f.write(f"Total Predictions Made By Model: {confusion_matrix.get_num_predictions()}\n")
        f.write(f"Total Number of Labels: {confusion_matrix.get_num_actual()}\n")
        f.write(f"Precision: {confusion_matrix.get_precision()}\n")
        f.write(f"Recall: {confusion_matrix.get_recall()}\n")
        f.write(
            f"IOU: {confusion_matrix.get_iou()}\n"
        )
