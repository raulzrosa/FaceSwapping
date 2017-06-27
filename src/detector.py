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

# TODO find multiple faces
def findFaces(gray):
    face = findLargest(gray)
    if face is None:
        return []
    else:
        return [face]


# TODO? face tracking

