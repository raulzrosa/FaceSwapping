#!/usr/bin/env python3

# Group 6
# Hugo Moraes Dzin 8532186
# Matheus Gomes da Silva Horta 8532321
# Raul Zaninetti Rosa 8517310


import cv2


# Returns a region of interest (ROI)
def roi(image, region):
    x, y, w, h = region
    return image[y:y+h, x:x+w]


# True if <rect1> and <rect2> overlap
def overlaps(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    overlaps_x = (x1 < x2 + w2) and (x2 < x1 + w1)
    overlaps_y = (y1 < y2 + h2) and (y2 < y1 + h1)

    return overlaps_x and overlaps_y


# Return the largest rectangle, or <rect1> if they have the same size
def largest(rect1, rect2):
    _, _, w1, h1 = rect1
    _, _, w2, h2 = rect2

    if w1*h1 >= w2*h2:
        return rect1
    else:
        return rect2


def drawRect(target, rect, color):
    x, y, w, h = rect
    cv2.rectangle(target, (x, y), (x+w, y+h), color, 1)
