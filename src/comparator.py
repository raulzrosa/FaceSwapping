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
    def __init__(self, index, face_bgr, face_gray, mouth_rect, nose_width, dist_eyes, hist, siftDesc):
        self.index = index
        self.face_bgr = face_bgr
        self.face_gray = face_gray

        self.mouth_rect = mouth_rect
        self.nose_width = nose_width
        self.dist_eyes = dist_eyes

        self.hist = hist
        self.siftDesc = siftDesc


def initData(search_dir):
    filenames = glob.glob(os.path.join(search_dir, '*.jpg'))

    test_data = []
    for index, f in enumerate(filenames):
        # print(f[len('professors.'):])
        bgr = cv2.imread(f)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        face = detector.findLargest(gray)

        roi_face = utils.roi(gray, face)

        if face is not None:
            cropped_bgr = utils.roi(bgr, face).copy()
            cropped_gray = utils.roi(gray, face).copy()
            hist = cv2.calcHist([cropped_gray], [0], None, [256], [0, 256])
            _, siftDesc = sift.detectAndCompute(cropped_gray, None)

            mouth_rect = detector.findMouth(roi_face)

            dist_eyes = detector.eyeDistance(roi_face)
            nose_width = detector.noseWidth(roi_face)

            test_data.append(FaceEntry(index, cropped_bgr, cropped_gray, mouth_rect, nose_width, dist_eyes, hist, siftDesc))

    return test_data


def calculateSimilarity(gray, test_data, comparator):
    final_value = 0
    most_equal = test_data[0].index

    if comparator == 'hist':
        hist_input = cv2.calcHist([gray], [0], None, [256], [0, 256])

        for professor in test_data:
            partial_value = cv2.compareHist(hist_input, professor.hist, cv2.HISTCMP_CORREL)
            if partial_value > final_value:
                final_value = partial_value
                most_equal = professor.index

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
            #print(len(good))
            partial_value = len(good)
            if partial_value > final_value:
                final_value = partial_value
                most_equal = professor.index

    return most_equal


# TODO
# test_data is a list of FaceEntry objects
# test_data = list with professors faces
# Select whitch algorithm to use by changing the third argument
def calculateMostSimilar(gray, test_data):
    best_sift = calculateSimilarity(gray, test_data, 'sift')
    scores = np.zeros(len(test_data), np.int)

    NOSE_SCORE = 30
    EYE_SCORE = 30
    SIFT_SCORE = 40

    nose_width = detector.noseWidth(gray)
    eye_dist = detector.eyeDistance(gray)

    if nose_width is not None:
        best_noses = top_nose_dif(nose_width, test_data)
        addScore(scores, best_noses, NOSE_SCORE)
    if eye_dist is not None:
        best_eyes = top_eye_dif(eye_dist, test_data)
        addScore(scores, best_eyes, EYE_SCORE)

    scores[best_sift] += SIFT_SCORE

    #print(best_noses)
    #print(best_eyes)
    #print(best_sift)

    best_index = np.argmax(scores)
    return test_data[best_index].face_bgr


WEIGHTS = list(range(1, 11))
# add up scores with weights
def addScore(total_scores, prof_results, multiplier, weights=WEIGHTS):
    for (index, (prof, _)) in enumerate(prof_results):
        total_scores[prof.index] += multiplier * weights[index]


# Get <top_count> professors with mose similar eye distance
def top_eye_dif(input_dist_eyes, test_data, top_count=10):
    # compute and sort by difference of eye_distance
    prof_scores = [
            (prof, abs(input_dist_eyes - prof.dist_eyes))
            for prof in test_data
            if prof.dist_eyes is not None
    ]

    # sorted_profs[0] is the worst, sorted_profs[-1] is the best
    sorted_profs = sorted(prof_scores, key=lambda x: x[1])

    # take only the <top_count> best
    best_matches = sorted_profs[-top_count:]

    return best_matches


# Get <top_count> professors with mose similar nose width
def top_nose_dif(input_nose_width, test_data, top_count=10):
    # compute and sort by difference of eye_distance
    prof_scores = [
            (prof, abs(input_nose_width - prof.nose_width))
            for prof in test_data
            if prof.nose_width is not None
    ]

    # sorted_profs[0] is the worst, sorted_profs[-1] is the best
    sorted_profs = sorted(prof_scores, key=lambda x: x[1])

    # take only the <top_count> best
    best_matches = sorted_profs[-top_count:]

    return best_matches
