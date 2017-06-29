#!/usr/bin/env python3

# Group 6
# Hugo Moraes Dzin 8532186
# Matheus Gomes da Silva Horta 8532321
# Raul Zaninetti Rosa 8517310

import numpy as np
import cv2

from src import utils


def cascadeFromXml(name):
    filename = 'haarcascades/' + name + '.xml'
    return cv2.CascadeClassifier(filename)


# https://github.com/opencv/opencv/tree/master/data/haarcascades
FACE_HAAR = cascadeFromXml('haarcascade_frontalface_default')
EYE_HAAR = cascadeFromXml('haarcascade_eye')

# http://alereimondo.no-ip.org/OpenCV/34
MOUTH_HAAR = cascadeFromXml('modesto_mouth')


def findLargest(gray):
    faces = FACE_HAAR.detectMultiScale(gray, 1.1, 3)

    if len(faces) == 0:
        return None  # No faces found

    face_area = [w*h for (_, _, w, h) in faces]
    largest_face = faces[np.argmax(face_area)]

    return largest_face


def findMouth(largest_face):
    mouths = MOUTH_HAAR.detectMultiScale(largest_face, 1.1, 3)
    if len(mouths) == 0:
        return None

    mouth_y = [y+h for (x, y, w, h) in mouths]
    lowest_mouth = mouths[np.argmax(mouth_y)]

    return lowest_mouth


def findEye(largest_face):
    eyes = EYE_HAAR.detectMultiScale(largest_face, 1.1, 3)
    if len(eyes) == 0:
        return None

    eyes_y = [y+h for (x, y, w, h) in eyes]
    lowest_eye = eyes[np.argmax(eyes_y)]

    return lowest_eye


def findFaces(gray):
    matches = FACE_HAAR.detectMultiScale(gray, 1.1, 6, minSize=(40, 40))

    good_matches = []
    bad_matches = []

    for face1 in matches:
        face1_is_good = True

        # Is there a larger face overlapping this one?
        # Yes -> face1 is no good
        for face2 in matches:
            if face1 is not face2:
                they_overlap = utils.overlaps(face1, face2)
                face2_larger = face2 is utils.largest(face1, face2)

                if they_overlap and face2_larger:
                    face1_is_good = False

        if face1_is_good:
            good_matches.append(face1)
        else:
            bad_matches.append(face1)

    return (good_matches, bad_matches)
