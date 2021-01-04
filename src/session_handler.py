import csv
import cv2
import numpy as np

class SessionHandler:
    def __init__(self, session_path):
        self.session_path = session_path
        self.data = self.__read()

    def get_data(self):
        return self.data

    def get_center_images(self):
        return self.data["center"]

    def get_left_images(self):
        return self.data["left"]

    def get_right_images(self):
        return self.data["right"]

    def get_measurements(self):
        return self.data["meas"]

    def __read(self):
        lines = {"center": [], "left": [], "right": [], "meas": []}
        with open(self.session_path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                center_img = cv2.cvtColor(cv2.imread(line[0]), cv2.COLOR_BGR2RGB)
                left_img = cv2.cvtColor(cv2.imread(line[1]), cv2.COLOR_BGR2RGB)
                right_img = cv2.cvtColor(cv2.imread(line[2]), cv2.COLOR_BGR2RGB)
                meas = float(line[3])
                lines["center"].append(center_img)
                lines["left"].append(left_img)
                lines["right"].append(right_img)
                lines["meas"].append(meas)
            lines["center"] = np.array(lines["center"])
            lines["left"] = np.array(lines["left"])
            lines["right"] = np.array(lines["right"])
            lines["meas"] = np.array(lines["meas"])
        return lines
