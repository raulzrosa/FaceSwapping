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
EYE_HAAR   = cascadeXml('haarcascade_eye')


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
    #if False:
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

            cropFace(roi_frame, (fx,fy,fw,fh))



    if False:
    #for face in faces:
        should_skip = False

        for other_face in faces:
            if (
                face is not other_face and
                overlaps(face, other_face) and
                area(face) < area(other_face)):
                    should_skip = True
                    break

        if should_skip:
            continue


        fx, fy, fw, fh = face
        roi_frame  = frame[fy:(fy+fh), fx:fx+fw]
        roi_gray = gray[fy:(fy+fh), fx:fx+fw]
        roi_gray = cv2.normalize(roi_gray, None, 0, 255, cv2.NORM_MINMAX)

        mouths = MOUTH_HAAR.detectMultiScale(roi_gray, 1.1, 3)
        for (mx, my, mw, mh) in mouths:
            cv2.rectangle(roi_frame, (mx, my), (mx+mw, my+mh), (0,0,255), 1)

        eyes = EYE_HAAR.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_frame, (ex, ey), (ex+ew, ey+eh), (255,0,255), 1)

        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0,255,0), 1)

        # Note: xfeatures2d needs opencv+contrib package
        #detector = cv2.xfeatures2d.SIFT_create(400)
        #kp = detector.detect(roi_gray, None)
        #cv2.drawKeypoints(roi_img, kp, roi_img)

    # Display the resulting frame
    showResized('Input Image', frame)


# Displays the cropped face on a new window
def cropFace(image, face):
    fx, fy, fw, fh = face

    # Crop face using elliptic mask

    # Determine ellipsis center and dimensions
    rows, cols, _ = image.shape
    height = int( np.round(rows * 0.9) )
    width  = int( np.round(cols * 0.75) )
    center = (cols // 2, rows // 2)
    dimensions = (width // 2, height // 2)

    # Create mask and inverse mask
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.ellipse(mask, center, dimensions, 0, 0, 360, (1,1,1), -1)
    anti_mask = np.ones(image.shape, dtype=np.uint8) - mask

    # Load the face that will be pasted
    new_face = loadNewFace()
    new_face = cv2.resize(new_face, image.shape[0:2])

    new_face = wrapper(new_face, image)


    # TODO color correction on skin tone

    # Paste image
    # The commented line does the same thing, but could be slower
    #image[:] = new_face * mask + image * anti_mask
    new_face *= mask
    image *= anti_mask
    image += new_face


def loadNewFace():
    frame = cv2.imread('simoes.jpg')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    faces = FACE_HAAR.detectMultiScale(gray, 1.1, 3)

    if len(faces) == 0:
        raise RuntimeError('No faces found')

    # Choose largest face
    face_area = [w*h for (x,y,w,h) in faces]
    fx, fy, fw, fh = faces[ np.argmax(face_area) ]

    return np.array(frame[fy:fy+fh, fx:fx+fw])

def hist_match(source, template):
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def wrapper(src, dst):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2YCR_CB)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2YCR_CB)

    channels = [None] * 3

    for i in range(3):
        channels[i] = hist_match(src[:,:,i], dst[:,:,i]).astype(np.uint8)

    return cv2.cvtColor(cv2.merge(channels), cv2.COLOR_YCR_CB2BGR)



def overlaps(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    overlap_x = x1 < x2 + w2 and x2 < x1 + w1
    overlap_y = y1 < y2 + h2 and y2 < y1 + h1

    return overlap_x and overlap_y

def area(rect):
    x, y, w, h = rect
    return w * h



if __name__ == '__main__':
    main()
