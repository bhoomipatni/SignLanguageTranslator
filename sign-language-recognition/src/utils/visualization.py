import cv2
import numpy as np

def draw_keypoints(image, keypoints, color=(0, 255, 0), radius=5):
    for point in keypoints:
        if point is not None:
            cv2.circle(image, (int(point[0]), int(point[1])), radius, color, -1)

def draw_gesture_label(image, label, position=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2):
    cv2.putText(image, label, position, font, font_scale, color, thickness)

def visualize(image, keypoints, label):
    image_copy = image.copy()
    draw_keypoints(image_copy, keypoints)
    draw_gesture_label(image_copy, label)
    return image_copy