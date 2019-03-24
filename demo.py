import numpy as np
import scipy as sp
import cv2
import os
import glob
import pickle
import MTCNN as mtcnn
import scipy

feature_file = np.load("data/feature_file.npz")
UNKNOWN_LABEL = "Unknown"
THRESHOLD = 0.75

def draw_label(image, point, label, emotion, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)
        cv2.putText(image, emotion, (point[0], point[1]+20), font, font_scale, (255, 255, 255), thickness)

label_array = []
feature_array = []
for key in feature_file:
    label_array.append(key)
    feature_array.append(feature_file[key])

def get_label(feature):
    distance = float("inf")
    label = None

    dist = scipy.spatial.distance.cdist(feature.reshape((1, feature.size)), np.array(feature_array), 'cosine')
    closest_index = np.argmin(dist)
    distance, label = dist[0][closest_index], label_array[closest_index] 

    return label if distance < THRESHOLD else UNKNOWN_LABEL 


def get_margins(face_margin, margin=1):
    (x, y, w, h) = face_margin[0], face_margin[1], face_margin[2] - face_margin[0], face_margin[3] - face_margin[1]
    margin = int(min(w, h) * margin / 100)
    x_a = int(x - margin)
    y_a = int(y - margin)
    x_b = int(x + w + margin)
    y_b = int(y + h + margin)
    return (x_a, y_a, x_b - x_a, y_b - y_a)


def face_demo():
    #video_capture = cv2.VideoCapture('/home/sachin/555.mp4')
    video_capture = cv2.VideoCapture(0)
    while True:
        if not video_capture.isOpened():
            sleep(5)
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        _, boundingboxes, features, emotion = mtcnn.process_image(frame)

        # placeholder for cropped faces
        for i in range(boundingboxes.shape[0]):
            (x, y, w, h) = get_margins(boundingboxes[i, 0:4])
            label = get_label(features[i]) if i < len(features) else UNKNOWN_LABEL
            #print len(features), i
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
            #label = "{}".format(label)
            #draw_label(frame, (x,y), "{}{}".format(label, emotion))
            draw_label(frame, (x,y), label, emotion)
    
        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(5) == 27 :  # ESC key press
            break
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    face_demo()
