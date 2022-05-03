from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import pickle
import argparse
import os

import cv2
from PIL import Image
import numpy as np
import pandas as pd
from FeatureExtraction import *

def get_test_images(images_dir):
    print("Reading the images from the dataset.")
    images = []
    correct_img = [0,1,4,6,9,10,11,13,18,19,20,21,22,23,25,27,28,29,31,32,33,36,37,38,41,44,45,46,49,50,53,55,57,60,61,
                    62,64,68,73,80,83,84,87,88,89,90,92,94,95,96,97,98,99,102,104,114,115,116,117,121,123,131,132,139,
                   140,141,142,143,149,150,151,153,157,158,159,160,162,163,164,173,176,182,183,200,203,205,207,208,209,213,216,217]
    for i in correct_img:
        image_file=os.listdir(images_dir)[i]
    #for image_file in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_file)
        image = Image.open(image_path)
        images.append(image)
    images = np.stack(images)
    return correct_img, images



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="lfpw", help='Name of the dataset.')
parser.add_argument('--dataset_dir', type=str, default="./datasets/lfpw/processed",
                    help='Directory of the dataset that the regressor is trained on.')
parser.add_argument('--model_save_dir', type=str, default="./saved_models",
                    help='Where the trained model will be saved.')
args = parser.parse_args()


# get the test directory
test_directory = os.path.join(args.dataset_dir, "test")
test_images_dir = os.path.join(test_directory, "images")
test_landmarks_path = os.path.join(test_directory, "landmarks.csv")

# load the model from disk
filename = os.path.join(args.model_save_dir, f"boosting_{args.dataset}.model")
boostingClassifier = pickle.load(open(filename, 'rb'))

# generate the Gabor filters
filter_bank = gabor_bank()

# get  images
test_nums, test_images = get_test_images(test_images_dir)

# get  landmarks
test_landmarks = pd.read_csv(test_landmarks_path, index_col=0)
# get no of landmarks
num_landmark_coordinates = len(test_landmarks.columns)
no_train_imgs = test_images.shape[0]

print(test_landmarks)

print("Testing has started!")

####################################  Testing  #####################################################
test_landmarks_np=np.zeros((len(test_nums),38))
# iterate over the training images
for i in range(len(test_nums)):
    # get the corresponding landmarks
    print(test_nums[i])
    landmarks = np.array(test_landmarks)[test_nums[i], 2:]
    test_landmarks_np[i] = landmarks



for t in range(len(test_images)):
    t_img=test_images[t]
    # convert to grayscale
    bw_img = cv2.cvtColor(t_img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("orig_img ", bw_img)
    #cv2.waitKey()
    (y_left, x_left), (y_right, x_right), (mouth_y, mouth_x)=find_initial_points(bw_img)
    ED = x_right-x_left

    ROIS = find_roi(bw_img, ED, (y_left, x_left), (y_right, x_right), (mouth_y, mouth_x))
    test_roi= ROIS[6]

    preds= np.zeros((25, 25))
    for i in range(6, 31):
        for j in range(6, 31):
            t_patch = test_roi[i-6:i+7, j-6:j+7]
            t_patch = np.reshape(t_patch, (1, np.size(t_patch)))
            pred = boostingClassifier.predict_proba(t_patch)
            preds[i-6, j-6] = pred[:, 1]

    try:
        min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(preds)
        max_pred = np.argmax(preds)
        p_x, p_y = max_indx
        p_x = p_x+6
        p_y = p_y + 6
        marked_roi = cv2.circle(test_roi, (p_y, p_x), radius=0, color=0, thickness=5)
    except:
        marked_roi=test_roi

    bw_img = cv2.circle(bw_img, (int(test_landmarks_np[t, 0]), int(test_landmarks_np[t, 1])), radius=0, color=255, thickness=5)
    cv2.imshow("original", bw_img)
    #cv2.imshow("predicted", marked_roi)
    cv2.waitKey()


