#!/usr/bin/env python3

# Group 6
# Hugo Moraes Dzin 8532186
# Matheus Gomes da Silva Horta 8532321
# Raul Zaninetti Rosa 8517310

import glob
import os.path

import cv2
import numpy as np

from src import utils, detector

class FaceEntry:
    def __init__(self, face_bgr, face_gray, mouth_bgr, mouth_gray, eye_bgr, eye_gray):
        self.face_bgr = face_bgr
        self.face_gray = face_gray
        self.mouth_bgr = mouth_bgr
        self.mouth_gray = mouth_gray
        self.eye_bgr = eye_bgr
        self.eye_gray = eye_gray

def initData(search_dir):
    filenames = glob.glob(os.path.join(search_dir, '*.jpg'))

    test_data = []
    for f in filenames:
        # print(f[len('professors.'):])
        bgr = cv2.imread(f)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        face = detector.findLargest(gray)
        mouth = detector.findMouth(gray)
        eye = detector.findEye(gray)

        if face is not None:
            cropped_bgr = utils.roi(bgr, face).copy()
            cropped_gray = utils.roi(gray, face).copy()
        if mouth is not None:
            cropped_mouth_bgr = utils.roi(bgr, mouth).copy()
            cropped_mouth_gray = utils.roi(gray, mouth).copy()

        if eye is not None:
            cropped_eye_bgr = utils.roi(bgr, eye).copy()
            cropped_eye_gray = utils.roi(gray, eye).copy()

            # TODO calculate and store metrics (SIFT, HOG etc)
            test_data.append(FaceEntry(cropped_bgr, cropped_gray, cropped_mouth_bgr, cropped_mouth_gray, cropped_eye_bgr, cropped_eye_gray))

    return test_data


#def HOG(gray, test_data):

# TODO
# test_data is a list of FaceEntry objects
# test_data = list with professors faces
def calculateMostSimilar(gray, test_data):
    return test_data[1].eye_bgr
