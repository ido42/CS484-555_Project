import argparse
import os
from PIL import Image
import numpy as np
import pandas as pd
import numpy as np

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

import pickle

from FaceDetection import *
from ExtractROIs import *
from FeatureExtraction import *
from train import *

from sklearn import metrics


def get_images(images_dir):
    print("Reading the images from the dataset.")
    images = []
    for image_file in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_file)
        image = Image.open(image_path)
        images.append(image)
    images = np.stack(images)
    return images


def save_model(model, model_save_path):
    with open(model_save_path, 'wb') as model_save_file:
        pickle.dump(model, model_save_file)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="lfpw", help='Name of the dataset.')
parser.add_argument('--dataset_dir', type=str, default="./datasets/lfpw/processed",
                    help='Directory of the dataset that the regressor is trained on.')
parser.add_argument('--model_save_dir', type=str, default="./saved_models",
                    help='Where the trained model will be saved.')
args = parser.parse_args()

# get the training directory
train_directory = os.path.join(args.dataset_dir, "train")
train_images_dir = os.path.join(train_directory, "images")
train_landmarks_path = os.path.join(train_directory, "landmarks.csv")

# get the test directory
test_directory = os.path.join(args.dataset_dir, "test")
test_images_dir = os.path.join(test_directory, "images")
test_landmarks_path = os.path.join(test_directory, "landmarks.csv")

# generate the Gabor filters
filter_bank = gabor_bank()


# get  images
train_images = get_images(train_images_dir)
test_images = get_images(test_images_dir)

# get  landmarks
train_landmarks = pd.read_csv(train_landmarks_path, index_col=0)
test_landmarks = pd.read_csv(test_landmarks_path, index_col=0)

print("Training has started!")

# get no of landmarks
num_landmark_coordinates = len(train_landmarks.columns)

# cv2.imshow("image", img)
# cv2.waitKey(0)

# define the used landmarks according to the paper (20 landmarks out of 68) - done in the preprocessing:LoadData.py file
# landmarks = [8, 17, 21, 22, 26, 30, 31, 35, 36, 37, 39, 41, 42, 44, 45, 46, 48, 51, 54, 57]
""" OLD CODE 
detect the faces
face_img = detect_face(imagePath="image.png")

#cv2.imshow("face", face_img)
#cv2.waitKey(0)

# percent by which the image is resized
scale_percent = 100

# calculate the 50 percent of original dimensions
width = int(face_img.shape[1] * scale_percent / 100)
height = int(face_img.shape[0] * scale_percent / 100)

# dsize
dsize = (width, height)

# resize image
face_img = cv2.resize(face_img, dsize)
"""


no_train_imgs = train_images.shape[0]

print(train_landmarks)

# define a list to store the features for all the training images
training_features = []

# iterate over the training images
for i in range(no_train_imgs):
    # get the each image
    face_img = train_images[i]

    # get the corresponding landmarks
    landmarks = np.array(train_landmarks)[i, :]

    # create an array to store the landmarks for further processing - p_landmarks
    p_landmarks = np.zeros((20, 2), dtype=np.int)
    i = 0
    for l_i in range(20):
        p_landmarks[l_i] = (landmarks[i], landmarks[i + 1])
        i = i + 2

    # print the landmarks for the ith image (to check them)
    print("p_land", p_landmarks)

    # convert to grayscale
    bw_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("orig_img ", bw_img)

###########################################  OLD CODE  must be put in a function ###############################################
    rows, cols = np.shape(bw_img)

    # bw_copy = bw_img.copy()
    # divide the picture into upper and lower halves
    upper_face = bw_img[0:int(np.ceil(rows / 2)), :]
    lower_face = bw_img[int(np.ceil(rows / 2)):, :]
    # cv2.imshow("upper", upper_face)
    # cv2.imshow("lower", lower_face)
    # cv2.waitKey(0)

    # divide the face into left and right (our left and right, not theirs)
    upper_left = upper_face[:, 0:int(np.ceil(cols / 2))]
    upper_right = upper_face[:, int(np.ceil(cols / 2)):]
    # cv2.imshow("upper left", upper_left)
    # cv2.imshow("upper right", upper_right)
    # cv2.waitKey(0)

    copy_left = upper_left.copy()
    copy_right = upper_right.copy()

    y_left, x_left = find_pupil(upper_left)
    y_right, x_right = find_pupil(upper_right)
    print(y_left, ", ", x_left)
    print(y_right, ", ", x_right)

    # the position of right pupil in the original image
    x_right_real = x_right + np.size(upper_left[0])

    # align the face so that the eye axis is parallel to horizon, if it is nearly parallel don't
    if abs(y_left - y_right) > 1:
        aligned = rotate_face(bw_img, (y_left, x_left), (y_right, x_right_real))
        upper_face = aligned[0:int(np.ceil(rows / 2)), :]
        upper_left = upper_face[:, 0:int(np.ceil(cols / 2))]
        upper_right = upper_face[:, int(np.ceil(cols / 2)):]
        y_left, x_left = find_pupil(upper_left)
        y_right, x_right = find_pupil(upper_right)
        print(y_left, ", ", x_left)
        print(y_right, ", ", x_right)
        # cv2.imshow("aligned", aligned)
        bw_img = aligned

    cv2.circle(copy_left, (x_left, y_left), radius=0, color=(0, 0, 255), thickness=5)
    cv2.circle(copy_right, (x_right, y_right), radius=0, color=(0, 255, 0), thickness=5)
    cv2.circle(face_img, (x_left, y_left), radius=0, color=(0, 0, 255), thickness=5)
    cv2.circle(face_img, (x_right_real, y_right), radius=0, color=(0, 0, 255), thickness=5)

    # cv2.imshow("eye l", copy_left)
    # cv2.imshow("eye r", copy_right)
    # cv2.imshow("eyes", bw_img)

    ED = x_right_real - x_left  # eye distance

    mouth_region = bw_img[y_left + int(0.85 * ED): y_left + int(1.5 * ED), x_left: x_right_real]
    # cv2.imshow("mouth region", mouth_region)
    # cv2.waitKey(0)

    # find edge and threshold the mouth region
    diff_ver = np.zeros((np.shape(mouth_region[:, 0])[0] - 1, np.shape(mouth_region[0])[0]))
    image_signed = np.array(mouth_region, dtype=np.int8)
    for i in range(np.shape(mouth_region[:, 0])[0] - 1):
        diff_ver[i] = np.abs(image_signed[i + 1].reshape((1, np.size(mouth_region[i + 1]))) -
                             image_signed[i].reshape((1, np.size(mouth_region[i]))))

    edge = diff_ver.copy()
    positions = np.where(diff_ver <= np.max(diff_ver) * 0.6)
    edge[diff_ver < np.max(diff_ver) * 0.2] = 0
    edge[diff_ver > np.max(diff_ver) * 0.2] = 255

    # cv2.imwrite("mouth edges.png", edge)
    # cv2.imshow("mouth edges", edge)
    # cv2.waitKey(0)

    vert_diff_summed = np.sum(diff_ver, axis=1)

    peaks, _ = find_peaks(vert_diff_summed)
    results_half = peak_widths(vert_diff_summed, peaks, rel_height=0.5)
    # plt.plot(np.array(range(0, np.size(vert_diff_summed))), vert_diff_summed)
    # plt.hlines(*results_half[1:], color="C2")
    # plt.title("Vertical Histogram of the Mouth")
    # plt.xlabel("Rows")
    # plt.ylabel("Pixel Intesity Differences Summed Over Columns")

    # plt.show()
    # plt.close()

    widest_peak = peaks[np.where(results_half[0] == np.max(results_half[0]))][0]
    mouth_y, mouth_x = int(widest_peak + y_left + 0.85 * ED), int((x_right_real + x_left) / 2)
    # cv2.circle(face_img, (mouth_x, mouth_y), radius=0, color=(255, 0, 0), thickness=5)
    # cv2.imshow("mouth ", face_img)
    # cv2.waitKey(0)

    # all_rois = find_roi(bw_img, ED, (y_left, x_left - 10), (y_right, x_right_real - 10), (mouth_y, mouth_x - 10))

    # cv2.imwrite("face 3 detected.png",face_img)

    # mouthlen_div3 = int((x_right_real-x_left)/5)
    # bw_copy = cv2.rectangle(bw_copy, (x_left-2*mouthlen_div3,int((y_left+0.85*ED+mouth_y)/2) ), (x_left+mouthlen_div3,int((y_left+1.5*ED+mouth_y)/2)), 255, 3)
    # bw_copy = cv2.rectangle(bw_copy, (x_right_real-2*mouthlen_div3,int((y_left+0.85*ED+mouth_y)/2) ), (x_right_real+mouthlen_div3,int((y_left+1.5*ED+mouth_y)/2)), 255, 3)
    # bw_copy = cv2.rectangle(bw_copy, (x_left-10,y_left-20 ), (int((x_right_real+x_left)/2)-10,y_left+20), 255, 3)

    # bw_copy = cv2.rectangle(bw_copy, (int(0.85*ED),int(x_left-2*mouthlen_div3)), (int(1.5*ED+mouth_y),int(mouth_x-mouthlen_div3)), 255, 3)
    # cv2.imshow("mouth rct ", bw_copy)
    # cv2.waitKey(0)

    # cv2.imwrite("roi detected.png",bw_copy)

####################################  Training examples generation #####################################################
    p_examples = np.zeros((20, 13, 13))  # positive patches 13x13
    n_examples = np.zeros((20, 13, 13))  # negative patches 13x13
    rois = np.zeros((20, 37, 37))  # 37x37 roi around the ground truth feature point
    for l in p_landmarks:
        # generate the examples based on the ground truth feature points
        p_examples[l], n_examples[l], rois[l] = get_training_examples(bw_img, l)


    # get the features using the ground truth rois
    train_img_features = feature_extraction(filter_bank, rois)

    # add the images features to the list of the features for all images
    training_features.append(train_img_features)


#################################### Training the regressor ###################################

# how to use the negative examples ???????????


# AdaBoostRegressor
# Choosing Decision Tree as the weak learner
DTR = DecisionTreeRegressor()
boostingClassifier = AdaBoostRegressor(n_estimators=50, base_estimator=DTR, learning_rate=1)

# Printing all the parameters of Adaboost
print(boostingClassifier)

# Creating the model on Training Data - we might need to restructure the input data
model_trained = boostingClassifier.fit(training_features, train_landmarks)

# save the model's weights
os.makedirs(args.model_save_dir, exist_ok=True)
model_save_path = os.path.join(args.model_save_dir, f"boosting_{args.dataset}.model")
save_model(boostingClassifier, model_save_path)
print("The model is saved to:", model_save_path)
print("Training is completed!")


####################################   Testing  #####################################################


## GET TEST FEATURES
# get rois

# get each 13x13 patch



#prediction=model_trained.predict(test_features)
# https://thinkingneuron.com/how-to-create-the-adaboost-regression-model-in-python/ the following code is from here

# Measuring Goodness of fit in Trainining data
print('R2 Value:', metrics.r2_score(train_landmarks, model_trained.predict(training_features)))

"""
# Measuring accuracy on Testing Data
print('Accuracy', 100 - (np.mean(np.abs((test_landmarks - prediction) / test_landmarks)) * 100))

# Plotting the feature importance for Top 10 most important columns

feature_importances = pd.Series(boostingClassifier.feature_importances_, index=Predictors)
feature_importances.nlargest(10).plot(kind='barh')

# Printing some sample values of prediction
TestingDataResults = pd.DataFrame(data=test_features, columns=Predictors)
TestingDataResults &  # 91;TargetVariable]=y_test
TestingDataResults &  # 91;('Predicted'+TargetVariable)]=prediction
TestingDataResults.head()

"""