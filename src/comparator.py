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

sift = cv2.xfeatures2d.SIFT_create()


class FaceEntry:
    def __init__(self, face_bgr, face_gray, mouth_bgr, mouth_gray, eye_bgr, eye_gray, hist, siftDesc):
        self.face_bgr = face_bgr
        self.face_gray = face_gray
        self.mouth_bgr = mouth_bgr
        self.mouth_gray = mouth_gray
        self.eye_bgr = eye_bgr
        self.eye_gray = eye_gray
        self.hist = hist
        self.siftDesc = siftDesc


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
            hist = cv2.calcHist([cropped_gray], [0], None, [256], [0, 256])
            _, siftDesc = sift.detectAndCompute(cropped_gray, None)

        if mouth is not None:
            cropped_mouth_bgr = utils.roi(bgr, mouth).copy()
            cropped_mouth_gray = utils.roi(gray, mouth).copy()

        if eye is not None:
            cropped_eye_bgr = utils.roi(bgr, eye).copy()
            cropped_eye_gray = utils.roi(gray, eye).copy()

        # TODO calculate and store metrics (SIFT, HOG etc)
        test_data.append(
            FaceEntry(cropped_bgr, cropped_gray, cropped_mouth_bgr, cropped_mouth_gray, cropped_eye_bgr,
                      cropped_eye_gray, hist, siftDesc))

    return test_data


# # histogram comparation
# def histComparison(gray, test_data):
#     hist_input = cv2.calcHist([gray], [0], None, [256], [0, 256])
#     final_value = 0
#     most_equal = test_data[0].face_bgr
#     for professor in test_data:
#         hist_professor = cv2.calcHist([professor.face_gray], [0], None, [256], [0, 256])
#         # testar outras metricas para computar
#         partial_value = cv2.compareHist(hist_input, hist_professor, cv2.HISTCMP_CORREL)
#         if partial_value > final_value:
#             final_value = partial_value
#             most_equal = professor.face_bgr
#     return most_equal


def calculateSimilarity(gray, test_data, comparator):
    final_value = 0
    most_equal = test_data[0].face_bgr

    if comparator == 'hist':
        hist_input = cv2.calcHist([gray], [0], None, [256], [0, 256])
        for professor in test_data:
            partial_value = cv2.compareHist(hist_input, professor.hist, cv2.HISTCMP_CORREL)
            if partial_value > final_value:
                final_value = partial_value
                most_equal = professor.face_bgr

    elif comparator == 'sift':
        kp1, des1 = sift.detectAndCompute(gray, None)
        good = []
        for professor in test_data:
            # BFMatcher with default params
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, professor.siftDesc, k=2)
            # Apply ratio test
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append([m])
            print(len(good))
            partial_value = len(good)
            if partial_value > final_value:
                final_value = partial_value
                most_equal = professor.face_bgr

    return most_equal


# def siftComparison(gray, test_data):
#     final_value = 0
#     most_equal = test_data[0].face_bgr
#     kp1, des1 = sift.detectAndCompute(gray, None)
#
#     for professor in test_data:
#         # find the keypoints and descriptors with SIFT
#         kp2, des2 = sift.detectAndCompute(professor.face_gray, None)
#
#         # BFMatcher with default params
#         bf = cv2.BFMatcher()
#         matches = bf.knnMatch(des1, des2, k=2)
#
#         # Apply ratio test
#         good = []
#         for m, n in matches:
#             if m.distance < 0.75 * n.distance:
#                 good.append([m])
#
#         partial_value = len(good)
#         if partial_value > final_value:
#             final_value = partial_value
#             most_equal = professor.face_bgr
#
#     return most_equal
#

# def HOG(gray, test_data):

# TODO
# test_data is a list of FaceEntry objects
# test_data = list with professors faces
# Select whitch algorithm to use by changing the third argument
def calculateMostSimilar(gray, test_data):
    most_similar = calculateSimilarity(gray, test_data, 'hist')
    # most_similar_hist_comp = histComparison(gray, test_data)
    return most_similar
