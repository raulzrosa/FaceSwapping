import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
img = cv2.imread('moacir.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
height, width, channels = img.shape
print ("ola pessoas",height, width, channels)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()