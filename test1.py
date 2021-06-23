import cv2
import numpy as np
import dlib
from math import hypot

import shapely
from shapely.geometry import LineString, Point

from pynput.mouse import Button, Controller

mouse = Controller()

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()

font = cv2.FONT_ITALIC

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)

    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    #Pupil position

    line1 = LineString([left_point, right_point])
    line2 = LineString([center_top, center_bottom])
    #if eye_points[0] == 36:
    int_pt = line1.intersection(line2)
    point_of_intersection = int_pt.x, int_pt.y
    #mouse.position = (int_pt.x, int_pt.y)
    print(eye_points[0])
    print(point_of_intersection)

    #a porjonto last edit

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    #print((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))


    ratio = hor_line_length / ver_line_length
    return ratio

def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)
                                ])

    #cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 3)
    #print(left_eye_region)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)

    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    # gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY_INV)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    gaze_ratio = left_side_white / right_side_white
    return gaze_ratio

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def midpoint(p1, p2):
        return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2) #cant be float

    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)

        #detect Blinking

        #left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)

        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)

        #blinking_ratio = (left_eye_ratio+right_eye_ratio) / 2

        #if blinking_ratio > 4.0:
            #cv2.putText(frame, "BLINKING", (50, 150), font, 3, (255, 0, 0))


        #gaze detection
        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        #cv2.putText(frame, str(gaze_ratio_left_eye), (200, 400), font, 2, (0, 0, 255), 3)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        #cv2.putText(frame, str(gaze_ratio_right_eye), (200, 450), font, 2, (0, 0, 255), 3)
        gaze_ratio1 = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2


        if gaze_ratio1 < 0.80:
            cv2.putText(frame, "Left", (50, 100), font, 2, (0, 0, 255), 3)
            #mouse.position = (30, 384)
        elif 0.80 < gaze_ratio1 < 1.5:
            cv2.putText(frame, "Center", (50, 100), font, 2, (0, 0, 255), 3)
            #mouse.position = (683, 384)
        else:
            cv2.putText(frame, "Right", (50, 100), font, 2, (0, 0, 255), 3)
            #mouse.position = (1330, 384)

        #cv2.putText(frame, str(left_side_white), (50, 100), font, 2, (0, 0, 255), 3)
        #cv2.putText(frame, str(right_side_white), (50, 150), font, 2, (0, 0, 255), 3)
        cv2.putText(frame, str(gaze_ratio1), (50, 150), font, 2, (0, 0, 255), 3)

        #threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
        #eye = cv2.resize(gray_eye, None, fx=5, fy=5)
        #cv2.imshow("Eye", eye)
        #cv2.imshow("Threshold", threshold_eye)
        #cv2.imshow("Left eye", left_eye)
        #cv2.imshow("Left Side Threshold", left_side_threshold)
        #cv2.imshow("Right Side Threshold", right_side_threshold)
        #cv2.imshow("Gaze Ratio", gaze_ratio)

        #x = landmarks.part(36).x
        #y = landmarks.part(36).y
        #cv2.circle(frame, (x, y), 3, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == ord("s"):
        break

cap.realease()
cv2.destroyAllWindows()