#!/usr/bin/env python3

# Group 6
# Hugo Moraes Dzin 8532186
# Matheus Gomes da Silva Horta 8532321
# Raul Zaninetti Rosa 8517310

import cv2
import numpy as np

import argparse as ap

from src import utils, detector, comparator, replacer


# Array of comparator.FaceEntry objects
TEST_DATA = []


def main():
    global TEST_DATA

    # Specify command-line arguments
    parser = ap.ArgumentParser(description='Replace faces in a picture.')
    parser.add_argument(
            '-f', '--filename', metavar='IMAGE', type=str,
            help='the image that wll be processed')
    parser.add_argument(
            '-d', '--directory', metavar='DIR', default='professors', type=str,
            help='directory containing replacement faces')
    parser.add_argument(
            '-m', '--mode', default='use-dir', type=str,
            help='MODE = use-dir: use replacement faces from -d. ' +
            'MODE = within-pic: swap the faces within input with eachother')

    args = parser.parse_args()

    if args.mode == 'use-dir':
        TEST_DATA = comparator.initData(args.directory)

    if args.filename is None:
        # No file specified, use camera
        fromCamera()
    else:
        image = cv2.imread(args.filename)
        processed = process(image)
        cv2.imshow('Result', processed)
        cv2.waitKey(0)


def fromCamera():
    camera = cv2.VideoCapture(0)

    while True:
        _, frame = camera.read()
        # Flip around Y axis to look more natural
        frame = cv2.flip(frame, 1)
        processed = test_hugo(frame)
        cv2.imshow('Video Capture', processed)

        ch = cv2.waitKey(1) & 0xFF
        if ch == ord('q'):
            break  # Close program
        if ch == ord('w'):
            pass  # TODO Compute most similar

    camera.release()


# Finds all faces in <bgr> and replaces each of them by another one
def process(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    faces = detector.findFaces(gray)

    for face in faces:
        roi_gray = utils.roi(gray, face)
        roi_bgr = utils.roi(bgr, face)

        new_face = comparator.calculateMostSimilar(roi_gray, TEST_DATA)
        replacer.pasteFace(roi_bgr, new_face)

    return bgr


def test_hugo(bgr, old_faces=[]):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    (faces, bad_matches) = detector.findFaces(gray)

    for face in faces:
        roi_gray = utils.roi(gray, face)
        roi_bgr = utils.roi(bgr, face)

        new_face = comparator.calculateMostSimilar(roi_gray, TEST_DATA)
        replacer.pasteFace(roi_bgr, new_face)

    return bgr


if __name__ == '__main__':
    main()
