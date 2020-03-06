import cv2
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from hog_svm import crop_plate
from skimage.feature import hog


clf = load('trainModel.joblib')
def detected_char(img) :
    plate = []
    img =cv2.resize(img,(280,300)) #(img.shape[1]*2, img.shape[0]*2))
    img_blur = cv2.medianBlur(img,7)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,15,2)

    # ret, thresh = cv2.threshold(img_gray,150,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((4,3), np.uint8)
    kernel2 = np.ones((5,5), np.uint8)
    kernel_dilate = np.ones((3,3), np.uint8)
    # thresh =cv2.dilate(thresh,kernel_dilate)
    # thresh = cv2.morphologyEx(thresh,cv2.MORPH_CROSS,kernel_dilate)
    # morp_thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE, kernel=kernel)
    morp_thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, kernel=kernel)
    morp_thresh =cv2.dilate(morp_thresh,kernel_dilate)
    contours, hierarchy = cv2.findContours(morp_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # print(hierarchy)
    # print(contours)
    digit_rect_ =[]
    for cont, hier in zip(contours, hierarchy[0]):
        if(hier[0] >0):
            x,y,w,h = cv2.boundingRect(cont)
            if (w<h and w>15 and h >70 and w <50 and h <150):
                # print(w, h)
                digit_rect_.append(cv2.boundingRect(cont))
    digit_rect = digit_rect_[:]
    sum_y = []
    for rect in digit_rect:
        sum_y.append(rect[3])
    mean = np.mean(sum_y)
    dist_h =[]
    # print(mean)
    i_=0

    # Remove all rect to big or to small
    while i_ <len(digit_rect):
        try:

            if (abs(digit_rect[i_][3]-mean)) > 10:
                err = digit_rect.pop(i_)

                # print(err)
                # print(i_)
            else:
                i_+=1
        except:
            print('i',i_)
            print(len(digit_rect))


    # Remove region out of plate
    for _ in range(2):
        try:

            if(abs(digit_rect[0][1] - digit_rect[1][1]) > 10):
                digit_rect.pop(0)
            if (abs(digit_rect[len(digit_rect) -1][1] - digit_rect[len(digit_rect)-2][1]) > 10):
                digit_rect.pop(len(digit_rect)-1)

        except:
            pass
            print('check')
    # prinh
    def custom_sort_x(elem):
        return elem[0]

    def custom_sort_y(elem):
        return elem[1]

    digit_rect.sort(reverse = False, key = custom_sort_y)
    digit_rect_above = digit_rect[0:4]
    digit_rect_above.sort(reverse = False, key = custom_sort_x)
    digit_rect_below = digit_rect[4:]
    digit_rect_below.sort(reverse = False, key = custom_sort_x)

    digit_rect = digit_rect_above + digit_rect_below

    for j,rect in enumerate(digit_rect):#[:len(digit_rect)-1]:
        x,y,w,h = rect
        roi = img[y:y+h, x:x+w]
        # print(roi)
        plate.append(roi)
        # print(roi.shape)

    pre_plate = []

    for char in plate :
        img = cv2.resize(char, (28,28))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _,img_thresh = cv2.threshold(img_gray,0,255,cv2.THRESH_OTSU)
        img_thresh = np.invert(img_thresh)
        img_train = img_thresh.reshape(784,)
        pre_plate.append(img_train)
        # feature = hog(img_thresh,orientations=9,pixels_per_cell=(3.5,3.5),cells_per_block=(2,2),block_norm="L2")
        # pre_plate.append(feature)

    # cv2.imshow('abc', plate[0])
    # cv2.waitKey(0)
    pre_plate = np.array(pre_plate)/255

    return pre_plate

# img = crop_plate('./GreenParking/img6.jpg')
# detected_char(img)