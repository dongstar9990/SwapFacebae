from builtins import print

import cv2
import dlib
import numpy as np

## Face detection
def face_detection(img,upsample_times=1):
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    detector = dlib.get_frontal_face_detector()
    faces = detector(img, upsample_times)
    print("face " , faces)
    return faces

PREDICTOR_PATH = 'models/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)
## Face and points detection
def face_points_detection(img, bbox:dlib.rectangle):
    # Get the landmarks/parts for the face in box d.
    shape = predictor(img, bbox)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    coords = np.asarray(list([p.x, p.y] for p in shape.parts()), dtype=int)
    print(coords)
    # return the array of (x, y)-coordinates
    return coords

def select_face(im, r=10, choose=True):
    faces = face_detection(im)
    if len(faces) == 0:
        print("1")
        return None, None, None

    if len(faces) == 1 or not choose:
        idx = np.argmax([(face.right() - face.left()) * (face.right() - face.top()) for face in faces])
        print(idx)
        bbox = faces[idx]
        print("2")
    else:
        bbox = []
        print("3")
        def click_on_face(event, x, y, flags, params):
            if event != cv2.EVENT_LBUTTONDOWN:
                return

            for face in faces:
                if face.left() < x < face.right() and face.top() < y < face.bottom():
                    bbox.append(face)
                    break

        im_copy = im.copy()
        for face in faces:
            # draw the face bounding box
            cv2.rectangle(im_copy, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 1)
        cv2.imshow('Click the Face:', im_copy)
        cv2.setMouseCallback('Click the Face:', click_on_face)
        while len(bbox) == 0:
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        bbox = bbox[0]

    points = np.asarray(face_points_detection(im, bbox))
    im_w, im_h = im.shape[:2]
    left, top = np.min(points, 0)
    # np.max(points, 0)
    right, bottom = np.max(points, 0)
    x, y = max(0, left - r), max(0, top - r)
    # max(0, left - r), max(0, top - r)
    w, h = min(right + r, im_h) - x,min( bottom + r, im_w) - y
        # min(right + r, im_h) - x,min( bottom + r, im_w) - y
    # print(f"left, top :{left} ,{top}")
    # print(f"weight img , heigh img : {im_w} {im_h}")
    # print(f"right bottom : {right} , {bottom}" )
    # print(f"w h {w} {h}")
    # print("x  y" , x ,y)
    # print("points ",points)

    # print("points ",points - np.asarray([[x, y]]), (x, y, w, h), im[y:y + h, x:x + w])
    return points - np.asarray([[x, y]]), (x, y, w, h), im[y:y + h, x:x + w]

def select_all_faces(im, r=10):
    faces = face_detection(im)

    if len(faces) == 0:
        return None

    faceBoxes = {k : {"points" : None,
                      "shape" : None,
                      "face" : None   } for k in range(len(faces))}
    for i, bbox in enumerate(faces):
        points = np.asarray(face_points_detection(im, bbox))

        im_w, im_h = im.shape[:2]
        left, top = np.min(points, 0)
        right, bottom = np.max(points, 0)
        # max(0, left - r), max(0, top - r)
        x, y = max(0, left - r), max(0, top - r)
        # min(right + r, im_h) - x, min(bottom + r, im_w) - y
        w, h =min(right + r, im_h) - x, min(bottom + r, im_w) - y
        faceBoxes[i]["points"] = points - np.asarray([[x, y]])
        faceBoxes[i]["shape"] = (x, y, w, h)
        faceBoxes[i]["face"] = im[y:y + h, x:x + w]
        # print("x y w h ", x,y,w,h)

    return faceBoxes
