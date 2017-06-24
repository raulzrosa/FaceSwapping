#!/usr/bin/env python3

# Group 6
# Hugo Moraes Dzin 8532816
# Matheus Gomes da Silva Horta 8532321
# Raul Zaninetti Rosa 8517310

import glob
import os.path

import cv2
import numpy as np

from src import utils, detector


class FaceEntry:
    def __init__(self, bgr, gray):
        self.bgr = bgr
        self.gray = gray


def initData(search_dir):
    filenames = glob.glob(os.path.join(search_dir, '*.jpg'))

    test_data = []
    for f in filenames:
        # print(f[len('professors.'):])
        bgr = cv2.imread(f)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        face = detector.findLargest(gray)

        if face is not None:
            cropped_bgr = utils.roi(bgr, face).copy()
            cropped_gray = utils.roi(gray, face).copy()

            # TODO calculate and store metrics (SIFT, HOG etc)
            test_data.append(FaceEntry(cropped_bgr, cropped_gray))

    return test_data


# TODO
# test_data is a list of FaceEntry objects
def calculateMostSimilar(gray, test_data):
    return test_data[0].bgr
