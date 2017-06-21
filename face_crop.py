# python webcam.py -> uses webcam
# python webcam.py -f FILENAME -> reads image from FILENAME

import argparse
import numpy as np
import cv2

def cascadeXml(name):
    filename = 'haarcascades/' + name + '.xml'
    return cv2.CascadeClassifier(filename)


FACE_HAAR  = cascadeXml('haarcascade_frontalface_default')
MOUTH_HAAR = cascadeXml('modesto_mouth')


# Draw a horizontal line at given x position
def horizontalLine(target, x, color, thickness=1):
    pt1 = (x, 0)
    pt2 = (x, target.shape[1])
    cv2.line(target, pt1, pt2, color, thickness)


# Resize image if it's too large, then displays
def showResized(window, image, max_height=700):
    resized = image

    (rows, cols, _) = image.shape
    if rows > max_height:
        factor  = max_height / rows
        resized = cv2.resize(image, (0,0), fx=factor, fy=factor)

    cv2.imshow(window, resized)


def main():
    # Reads command-line arguments (argc/argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, help='processes a picture from the specified file')
    args = parser.parse_args()

    if args.filename is None:
        fromCamera()
    else:
        image = cv2.imread(args.filename)
        process(image)
        cv2.waitKey(0)


def fromCamera():
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        process(frame)

        ch = cv2.waitKey(1) & 0xFF
        if ch == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Faz os paranaues na imagem <frame>
def process(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)


    faces = FACE_HAAR.detectMultiScale(gray, 1.1, 3)
    if len(faces) != 0:
        # Choose largest face
        face_area = [w*h for (x,y,w,h) in faces]
        fx, fy, fw, fh = faces[ np.argmax(face_area) ]

        # Set region of interest (face region)
        roi_frame  = frame[fy:(fy+fh), fx:fx+fw]
        roi_gray = gray[fy:(fy+fh), fx:fx+fw]
        roi_gray = cv2.normalize(roi_gray, None, 0, 255, cv2.NORM_MINMAX)


        mouths = MOUTH_HAAR.detectMultiScale(roi_gray, 1.1, 3)
        if len(mouths) != 0:
            # Choose lowest mouth
            bottom_y = [y+h for (x, y, w, h) in mouths]
            mx, my, mw, mh = mouths[ np.argmax(bottom_y) ]

            roi_copy = np.array(roi_frame)
            cv2.rectangle(roi_frame, (mx, my), (mx+mw, my+mh), (0,0,255), 2)

            center_x = mx + mw // 2
            horizontalLine(roi_frame, center_x, (0,0,255))

            cropFace(roi_copy, (fx,fy,fw,fh), (mx,my,mw,mh))

        # Note: xfeatures2d needs opencv+contrib package
        #detector = cv2.xfeatures2d.SIFT_create(400)
        #kp = detector.detect(roi_gray, None)
        #cv2.drawKeypoints(roi_img, kp, roi_img)

        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0,255,0), 2)

        center_x = fx + fw // 2
        horizontalLine(frame, center_x, (0,255,0))

    # Display the resulting frame
    #frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    showResized('Input Image', frame)


# Displays the cropped face on a new window
def cropFace(image, face, mouth):
    fx, fy, fw, fh = face
    mx, my, mw, mh = mouth

    #new_w = fw * 0.75
    #new_x = int(center_x - new_w/2)
    #new_w = int(new_w)
    #crop_region = frame[fy : fy+fh,  new_x : new_x+new_w]

    mask = np.zeros(image.shape, dtype=np.uint8)

    crop_region = image
    rows, cols, _ = image.shape
    width = int( np.round(cols * 0.75) )

    center = (cols // 2, rows // 2)
    axes = (width // 2, rows // 2)

    cv2.ellipse(mask, center, axes, 0, 0, 360, (1,1,1), -1)

    image *= mask

    cv2.imshow('Crop Region', cv2.resize(crop_region, (300, 300)))



if __name__ == '__main__':
    main()
