import os 
import cv2 
import dlib 

import numpy as np 
import itertools as it 
import functools as ft 

import json, pickle, joblib 
import argparse 

from os import path 
from glob import glob

import torch as th 
import torchvision as tv 
from torchvision import transforms as T 


def define_window(window_name, width, height):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

def pause():
    cv2.waitKey()

def destroy():
    cv2.destroyAllWindows()

def read_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR)

def resize(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def display(target_window, image):
    cv2.imshow(target_window, image)

def get_face_detector():
    return dlib.get_frontal_face_detector()

def get_shape_predictor(shape_predictor_path):
    return dlib.shape_predictor(shape_predictor_path)

def get_face_recognizer(face_recognition_path):
    return dlib.face_recognition_model_v1(face_recognition_path)

def find_face_regions(rgb_image, detector):
    return detector(rgb_image, 1)


def get_coordinates(boxe):
    return np.array( [boxe.left(), boxe.top(), boxe.right(), boxe.bottom()] )


def get_landmarks(rgb_image, boxe, shape_predictor):
    return shape_predictor(rgb_image, boxe)


def get_128d_embeddings(rgb_image, landmarks, face_recognizer):
    return np.asarray(face_recognizer.compute_face_descriptor(rgb_image, landmarks, 10, 0.25))

def map_coordinates(W0, H0, W1, H1, x, y):
    a, b = W1 / W0, H1 / H0 
    return np.round( np.array([a * x, b * y]) ).astype('int')


def extract_face(image, coordinates):
    [x0, y0, x1, y1] = coordinates
    return image[y0:y1, x0:x1, :].copy()


def draw_face_region(image, coordinates):
    x0, y0, x1, y1 = coordinates
    cv2.rectangle(image, (x0, y0), (x1, y1), (255, 0, 0), 2)

def write_text(image, text, x, y):
    (w, h), tb = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
    cv2.rectangle(image, (x, y), (x + w, y - h - 2 * tb), (255, 255, 255), -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)

def write_texts(image, texts, x, y, font_scale, thickness):
    top_h = 0
    top_w = 0
    acc = []
    for t in texts:
        (w, h), tb = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        acc.append([t, h, tb])
        top_w = max([top_w, w])
        top_h = top_h + h + 2 * tb 
    cv2.rectangle(image, (x, y), (x + top_w, y - top_h), (255, 255, 255), -1)
    
    for i, (t, h, tb) in enumerate(acc):
        cv2.putText(image, t, (x, y - i * (h + 2 * tb)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

def capture(video_file_path):
    return cv2.VideoCapture(video_file_path)

def video_writer(video_output_path, width, height):
    four_cc = cv2.VideoWriter_fourcc(*'DIVX')
    return cv2.VideoWriter(video_output_path, four_cc, 30, (width, height), True)

def build_interval(response):
    names, value, prob = response 
    _, _, m, n = names.split('/')[-1].split('.')[0].split('_')
    m = int(m)
    n = int(n)
    a = (n - m) // 2
    b = (n + m) // 2
    prob = th.squeeze(prob).numpy()
    f = int(b - prob[1] * a)
    g = int(b + prob[0] * a)
    return f, g

def recursive_prediction(names, models, prd, idx, cnt, features, crr, prob):
    if cnt < 4:
        mdl = models[idx]
        with th.no_grad():  
            tensor_features = th.from_numpy(features).float()
            predicted_age = th.softmax(mdl(tensor_features[None, :]), dim=1)
        age = th.argmax(predicted_age, dim=1)
        return recursive_prediction(names, models, idx, 2 * idx + 1 + age[0], cnt + 1, features, age[0], predicted_age)
    else:
        return names[prd], crr.item(), prob 