import cv2
import numpy as np
from sklearn import svm, metrics
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
from skimage import io
import glob
import os
from skimage import img_as_ubyte
from joblib import load


# bo
def get_images(num_file, number):
    f = open(num_file, "rb") # Open file in binary mode
    # g = open(let_file, "rb")
    f.read(16) # Skip 16 bytes header
    # g.read(16)
    images = []

    for i in range(number):
        image = []
        for j in range(28*28):
            image.append(ord(f.read(1)))
            # image.append(ord(g.read(1)))
        images.append(image)
    return images

def get_labels(num_file, number):
    l = open(num_file, "rb")
    # k = open(let_file, "rb") # Open file in binary mode
    l.read(8) # Skip 8 bytes header
    # k.read(8)
    labels = []
    for i in range(number):
        labels.append(ord(l.read(1)))
        # labels.append(ord(k.read(1)))
    return labels

# def convert_png(images, labels, directory):
#     if not os.path.exists(directory):
#         os.mkdir(directory)

#     for i in range(len(images)):
#         out = os.path.join(directory, "%06d-num%d.png"%(i,labels[i]))
#         io.imsave(out, np.array(images[i]).reshape(28,28))


# number = 10000
# train_images = get_images("./emnist/emnist-letters-train-images-idx3-ubyte", number)
# train_labels = get_labels("./emnist/emnist-letters-train-labels-idx1-ubyte", number)

# convert_png(train_images, train_labels, "train")


from mnist import MNIST
mnist_data = MNIST('./mnist')
# train_images, train_labels = mnist_data.load_training()
test_images, test_labels = mnist_data.load_testing()


# TRAINING_SIZE = 5000
# train_images = get_images("./emnist/emnist-letters-train-images-idx3-ubyte", TRAINING_SIZE)
# train_images = np.array(train_images)/255
# train_labels = get_labels("./emnist/emnist-letters-train-labels-idx1-ubyte", TRAINING_SIZE)

clf = load('trainModel.joblib')


# clf = svm.SVC(C=100)
# clf.fit(train_images, train_labels)

# dump(clf, 'trainModel.joblib')

TEST_SIZE = 500
test_images = get_images("./mnist/t10k-images-idx3-ubyte", TEST_SIZE)
test_images = np.array(test_images)/255
test_labels = get_labels("./mnist/t10k-labels-idx1-ubyte", TEST_SIZE)

# im = cv2.imread("./image/chars/chars2_7.jpg")
# im = cv2.resize(im, (28,28), interpolation = cv2.INTER_AREA)
# im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# img = np.invert(im_gray)
# img = img.reshape(-1, 784)
# # print(img.shape)


predict = clf.predict(test_images)
ac_score = metrics.accuracy_score(test_labels, predict)
print(ac_score)






