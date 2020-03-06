import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd
import time
import pickle
import os
from sklearn.metrics import confusion_matrix
import pandas as pd
import pickle
from scipy.ndimage.measurements import label

import warnings
warnings.filterwarnings("ignore")
#create data training from file Location that contain namefile and position of lisencse plate
def createData(datapath):
    with open(datapath + 'location.txt', encoding='utf-8') as f:
        l = f.read().split('\n')
        lines = [line.split(' ') for line in l]
        count_no_plate = 31
        for i, line_ in enumerate(lines):
            name = datapath + line_[0]
            img = cv2.imread(name)
            # print(name)
            pos = [int(po) for po in line_[2:]]
            # print(pos)
            #create
            img_crop = img[pos[1]:pos[1] + pos[3], pos[0]:pos[2] + pos[0], :]  # [y,x]
            cv2.imwrite('./image/plate/' + str(i) + '.jpg', img_crop)




            try:
                img_crop = img[pos[1]:pos[1] + pos[3], pos[0]-pos[2]:pos[0], :]
                cv2.imwrite('./image/no_plate/' + str(count_no_plate) + '.jpg', img_crop)
                count_no_plate +=1
            except:
                pass

            try:
                img_crop = img[pos[1]:pos[1] + pos[3], pos[0] + pos[2]:pos[0] + pos[2] +pos[2], :]
                cv2.imwrite('./image/no_plat/' + str(count_no_plate) + '.jpg', img_crop)
                count_no_plate +=1

            except:
                pass
            try:
                img_crop = img[pos[1] - pos[3]:pos[1], pos[0]:pos[2] + pos[0], :]  # [y,x]
                cv2.imwrite('./image/no_plat/' + str(count_no_plate) + '.jpg', img_crop)
                count_no_plate +=1

            except:
                pass

            try:
                img_crop = img[pos[1] + pos[3]:pos[1] + pos[3]+pos[3], pos[0]:pos[2] + pos[0], :]  # [y,x]
                cv2.imwrite('./image/no_plat/' + str(count_no_plate) + '.jpg', img_crop)
                count_no_plate +=1

            except:
                pass


# createData('D:/Machine_learning/data/GreenParking/') # chay đoạn này để crop ảnh biển số để train==> vào file download_file chạy func: conver_2_gray


pathdata = './image/grayimg/'
orient = 12
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
# Spatial size and histogram parameters
spatial_size=(9, 10)
hist_bins=16

# plate_names = os.listdir(pathdata + 'plate/')
# noplate_names = os.listdir(pathdata + 'no_plate/')


# Define HOG parameters
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


def extract_features(imgs, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, vis=False, type='T'):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        if(type=='T'):
            image = mpimg.imread(pathdata + 'plate/' + file)
        else:
            image = mpimg.imread(pathdata + 'no_plate/' + file)
        # apply color conversion to YCrCb
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

        # Call get_hog_features() with vis=False, feature_vec=True

        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features



def trainModel():
    print ('Extracting car features')
    plat_features = extract_features(plate_names, orient=orient,
                                     pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                     hog_channel=hog_channel, type='T')
    print ('Extracting not-car features')
    notplat_features = extract_features(noplate_names, orient=orient,
                                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                        hog_channel=hog_channel, type='F')

    # Create an array stack of feature vectors
    X = np.vstack((plat_features, notplat_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    print(scaled_X.shape)
    # Define the labels vector
    y = np.hstack((np.ones(len(plat_features)), np.zeros(len(notplat_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    print(X_train[0].shape)

    # print('Using:',spatial_size, 'spatial_size' , hist_bins, 'hist_bins')
    print('HOG: Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))


    # Use a linear SVC
    svc = LinearSVC()

    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # Check the prediction time for a single sample
    t=time.time()
    prediction = svc.predict(X_test[0].reshape(1, -1))
    t2 = time.time()
    print(t2-t, 'Seconds to predict with SVM')

    # Visualize a confusion matrix of the predictions
    pred = svc.predict(X_test)
    cm = pd.DataFrame(confusion_matrix(pred, y_test))
    print(cm)

    # Save a dictionary into a pickle file.

    classifier_info = { "svc": svc, "scaler": X_scaler, "orient": orient, "pix_per_cell": pix_per_cell,
    "cell_per_block": cell_per_block}

    pickle.dump( classifier_info, open( "classifier_info.p", "wb" ) )



# Define a single function that can extract features using hog sub-sampling and make predictions
def find_plates(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block):
    draw_img = np.copy(img)
    draw_img_all_windows = np.copy(img)
    img = img.astype(np.float32) / 255

    bbox_list = []

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    nfeat_per_block = orient * cell_per_block ** 2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = [90,81]
    nX_blocks_per_window = (window[0] // pix_per_cell) - 1
    nY_blocks_per_window = (window[1] // pix_per_cell) - 1

    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nX_blocks_per_window) // cells_per_step
    nysteps = (nyblocks - nY_blocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    found = False
    xb=0
    nxsteps_ = nxsteps
    nysteps_ = nysteps
    scores = []
    while xb < nxsteps_:
        yb=0
        hog_features = []
        while yb < nysteps_:
        # for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nY_blocks_per_window, xpos:xpos + nX_blocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nY_blocks_per_window, xpos:xpos + nX_blocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nY_blocks_per_window, xpos:xpos + nX_blocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell


            # Scale features and make a prediction
            test_features = X_scaler.transform(hog_features.reshape(1, -1))

            test_prediction = svc.decision_function(test_features)


            # test_prediction = svc.predict(test_features)
            print(test_prediction)

            xbox_left = np.int(xleft * scale)
            ytop_draw = np.int(ytop * scale)
            win_draw = [np.int(a * scale) for a in window]
            scores.append(test_prediction)
            if test_prediction > 0.0:
                print('check found')
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = [np.int(a * scale) for a in window]
                bbox_list.append(
                    ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw[0], ytop_draw + win_draw[1] + ystart)))
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw[0], ytop_draw + win_draw[1] + ystart), (0, 0, 255), 6)
                found = True
                if(xb+2 <= nxsteps):
                    nxsteps_ = xb+2
                if(yb+2 <= nysteps):
                    nysteps_ = yb +2
                # print(xb,'--',yb)
                # print('check-found: ', test_prediction)
            yb +=1

        xb += 1
    print(scores)
    return bbox_list, draw_img, draw_img_all_windows, found


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected plate
    for plate_number in range(1, labels[1] + 1):
        # Find pixels with each plate_number label value
        nonzero = (labels[0] == plate_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Crop ROI
        imgc = np.copy(img)
        roi = imgc[bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0]]
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img, roi


# trainModel() 

import pickle
# Load info
dist_pickle = pickle.load( open("classifier_info.p", "rb" ) )
svc_l = dist_pickle["svc"]
X_scaler_l = dist_pickle["scaler"]
orient_l = dist_pickle["orient"]
pix_per_cell_l = dist_pickle["pix_per_cell"]
cell_per_block_l = dist_pickle["cell_per_block"]


# Read an image to test
def detect_plate(img):
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    # Define ROI of the images where to use the sliding windows
    ystart = 10
    ystop = img.shape[0] - 10

    # Look for cars at different scales
    scales = np.arange(0.5, 2, 0.2)
    find = False
    for scale in scales:
        box_list, out_img, out_img_windows, find_ = find_plates(img, ystart, ystop, scale, svc_l, X_scaler_l, orient_l,
                                                                pix_per_cell_l, cell_per_block_l)
        if find_ == True:
            find = True
        heat = add_heat(heat, box_list)

    if find == True:
        print('found plate: ' + name + '--' + str(i) + ' scale: ' + str(scale))
        # break
    else:
        print('Not Found: ' + name + '--' + str(i))
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)  # connect region - find many plate in an image
    draw_img, roi = draw_labeled_bboxes(np.copy(img), labels, i)
    print('==>Detected')

# chạy đoạn này để crop ra ảnh biển số
def crop_plate(path) :
    img = cv2.imread(path)
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    # Define ROI of the images where to use the sliding windows
    ystart = 10
    ystop =img.shape[0]-10

    # Look for cars at different scales
    scales = np.arange(0.5,1.2,0.1)
    find =False
    for scale in scales:
        box_list, out_img, out_img_windows, find_ = find_plates(img, ystart, ystop, scale, svc_l, X_scaler_l, orient_l,
                                                        pix_per_cell_l, cell_per_block_l)
        if find_ ==True:
            find = True
        heat = add_heat(heat, box_list)
        # plt.imshow(out_img)
        # plt.show()
    heat = apply_threshold(heat,10)

    # Visualize the heatmap when displaying55
    heatmap = np.clip(heat, 0, 255)

    # plt.imshow(heatmap, cmap='viridis')
    # plt.show()
    # Find final boxes from heatmap using label function
    labels = label(heatmap) # connect region - find many plate in an image
    draw_img, roi = draw_labeled_bboxes(np.copy(img), labels)
    # plt.imshow(draw_img)
    # plt.show()
    return roi
   
crop_plate("D:\Machine_learning\data\GreenParking/0010_00004_b.jpg")
