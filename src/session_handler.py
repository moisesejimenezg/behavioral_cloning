import csv
import cv2
import numpy as np

class SessionHandler:
    def __init__(self, session_path):
        self.session_path = session_path
        self.data = self.__read()

    def get_data(self):
        return self.data

    def get_left_images(self):
        return self.data["left"]

    def get_right_images(self):
        return self.data["right"]

    def get_measurements(self):
        return self.data["meas"]

    def __read(self):
        lines = {"left": [], "right": [], "meas": []}
        with open(self.session_path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines["left"].append(cv2.imread(line[0]))
                lines["right"].append(cv2.imread(line[1]))
                lines["meas"].append(float(line[3]))
            lines["left"] = np.array(lines["left"])
            lines["right"] = np.array(lines["right"])
            lines["meas"] = np.array(lines["meas"])
        return lines
