import cv2
import numpy as np
from skimage.feature import hog
# from sklearn.svm import LinearSVC
# from sklearn import svm, metrics
from sklearn.metrics import accuracy_score
from skimage import io
import glob
import os
from joblib import dump, load
from hog_svm import crop_plate
from separate_char import detected_char

def predict_plate(path) :
    plate = crop_plate(path)
    char_list = detected_char(plate)

    clf = load('trainModel.joblib')
    predict = clf.predict(char_list)
    print(predict)

    img_plate = cv2.imread(path)
    cv2.imshow('plate', img_plate)
    cv2.waitKey(0)



predict_plate('./image/1.jpg')

