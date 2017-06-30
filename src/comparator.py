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

def cascadeFromXml(name):
    filename = 'haarcascades/' + name + '.xml'
    return cv2.CascadeClassifier(filename)

EYE_HAAR = cascadeFromXml('haarcascade_eye')
NOSE_HAAR = cascadeFromXml('Nariz')


class FaceEntry:
    def __init__(self, face_bgr, face_gray, mouth_bgr, mouth_gray, nose_bgr, nose_gray, dist_eyes, hist, siftDesc):
        self.face_bgr = face_bgr
        self.face_gray = face_gray
        self.mouth_bgr = mouth_bgr
        self.mouth_gray = mouth_gray
        self.nose_bgr = nose_bgr
        self.nose_gray = nose_gray
        
        self.dist_eyes = dist_eyes

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
        nose = detector.findNose(gray)

        dist_eyes = detector.findEye(gray)

        if face is not None:
            cropped_bgr = utils.roi(bgr, face).copy()
            cropped_gray = utils.roi(gray, face).copy()
            hist = cv2.calcHist([cropped_gray], [0], None, [256], [0, 256])
            _, siftDesc = sift.detectAndCompute(cropped_gray, None)

        if mouth is not None:
            cropped_mouth_bgr = utils.roi(bgr, mouth).copy()
            cropped_mouth_gray = utils.roi(gray, mouth).copy()

        if nose is not None:
            cropped_nose_bgr = utils.roi(bgr, nose).copy()
            cropped_nose_gray = utils.roi(gray, nose).copy()

        # TODO calculate and store metrics (SIFT, HOG etc)
        test_data.append(
            FaceEntry(cropped_bgr, cropped_gray, cropped_mouth_bgr, cropped_mouth_gray, cropped_nose_bgr, cropped_nose_gray, dist_eyes, hist, siftDesc))

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

def calc_input_dist_eyes(gray):
    eyes = EYE_HAAR.detectMultiScale(gray, 1.1, 3)
    if len(eyes) == 2:
        input_eyes = []
        for (x, y, w, h) in eyes:
            input_eyes.append([int(x + (w/2)),int(y + (h/2))])
        dist_eyes = input_eyes[0][0] - input_eyes[1][0]
        return dist_eyes
    else:
        return None

def top_eye_dif(gray, test_data):
    #10 melhores distancia entre olhos
    top10 = []
    input_dist_eyes = calc_input_dist_eyes(gray)

    for professor in test_data:
        #diference between input distance eyes and professor distance eyes
        if professor.dist_eyes != None:
            dif = abs(input_dist_eyes - professor.dist_eyes)
            if len(top10) < 10:
                top10.append([professor, dif])
            else:
                for top10_professor in top10:
                    if top10_professor[1] > dif:
                        top10.remove(top10_professor)
                        top10.append([professor, dif])
    sorted(top10, key= lambda x: x[1])
    return top10

def calc_input_nose_size(gray):
    noses = NOSE_HAAR.detectMultiScale(gray, 1.1, 3)
    if len(noses) == 0:
        return None

    nose_y = [y+h for (x, y, w, h) in noses]
    largest_nose = noses[np.argmax(nose_y)]
    cropped_nose_gray = utils.roi(gray, largest_nose).copy()
    return cropped_nose_gray


def top_nose_dif(gray, test_data):
    #10 melhores tamanhos de nariz
    top10 = []
    input_nose = calc_input_nose_size(gray)
    input_height, input_width = input_nose.shape

    for professor in test_data:
        prof_height, prof_width = professor.nose_gray.shape

        dif = abs(input_height - prof_height)
        if len(top10) < 10:
            top10.append([professor, dif])
        else:
            for top10_professor in top10:
                if top10_professor[1] > dif:
                    top10.remove(top10_professor)
                    top10.append([professor, dif])
    sorted(top10, key= lambda x: x[1])
    return top10


# TODO
# test_data is a list of FaceEntry objects
# test_data = list with professors faces
# Select whitch algorithm to use by changing the third argument
def calculateMostSimilar(gray, test_data):
    most_similar = calculateSimilarity(gray, test_data, 'sift')
    # most_similar_hist_comp = histComparison(gray, test_data)
    top10_nose = top_nose_dif(gray, test_data)
    top10_eye = top_eye_dif(gray, test_data)
    #most_similar = top10[9]

    #print(top10[9][0].face_bgr)
    return top10_eye[9][0].face_bgr




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