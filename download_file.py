import urllib.request as req
import cv2
import numpy as np
import  os

path_save = './image/'
path_pos = path_save+'positive/'
path_neg = path_save + 'negative/motor/'

if not os.path.exists(path_pos):
    os.makedirs(path_pos)
if not os.path.exists(path_neg):
    os.makedirs(path_neg)

def store_raw_image(listpath):
    with open(listpath, encoding='utf-8') as f:
        paths = f.read().split('\n')
        print(len(paths))
    for i,link in enumerate(paths[741:]):
        try:
            url = req.urlretrieve(link,path_neg+'neg_'+str(i+740)+'.jpg')
            print(i)
        except:
            pass

def conver_2_gray(path):
    if not os.path.exists(path_save+ 'grayimg/' + path):
        os.makedirs(path_save+ 'grayimg/' + path)
    for i, img_name in enumerate(os.listdir(path_save+ path)):
        # print(img_name,'--',i)
        img =cv2.imread(path_save + path + img_name)
        img_s = cv2.resize(img,(90,81))
        cv2.imwrite(path_save + 'grayimg/' + path +str(i)+'.jpg',img_s)


def create_pos_n_neg():
    for file_type in ['neg']:

        for img in os.listdir(path_neg+'grayimg/'):

            # if file_type == 'pos':
            #     line = file_type + '/' + img + ' 1 0 0 50 50\n'
            #     with open('info.dat', 'a') as f:
            #         f.write(line)
            # elif file_type == 'neg':
            line = path_neg + 'grayimg/' + img + '\n'
            with open('bg.txt', 'a') as f:
                f.write(line)

# store_raw_image('./motor.txt')
conver_2_gray('plate/') # chạy cái này để resize ảnh training
# create_pos_n_neg()