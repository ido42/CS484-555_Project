import argparse
import os

import cv2
from PIL import Image
import numpy as np
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import pickle

from FaceDetection import *
from ExtractROIs import *
from FeatureExtraction import *
from train import *

from sklearn import metrics


def get_train_images(images_dir):
    print("Reading the images from the dataset.")
    images = []
    correct_img = [0, 1, 5, 9, 11, 12, 13, 14, 16, 17, 19, 21, 23, 24, 28, 29, 31, 32, 34, 35, 39, 40, 41, 45, 46, 49,
                   51, 53, 54, 56, 57, 58, 59, 60, 61, 62,
                   63, 64, 67,
                   68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 79, 82, 84, 85, 87, 91, 93, 95, 96, 103, 105, 107, 110, 112,
                   114, 115, 116, 118, 120, 121, 123, 125, 127,
                   129, 133, 134, 135, 137, 139, 141, 143, 146, 148, 151, 155, 156, 158, 159, 161, 165, 168, 169, 170,
                   171, 173, 176, 177, 180, 181, 184, 186, 188,
                   189, 191, 193, 194, 195, 196, 199, 200, 201, 202, 203, 204, 205, 208, 209, 211, 212, 213, 214, 215,
                   217, 219, 223, 224, 225, 226, 230, 232, 233,
                   235, 237, 238, 239, 241, 245, 246, 250, 253, 255, 258, 259, 261, 262, 264, 265, 266, 267, 274, 275,
                   276, 278, 279, 281, 282, 286, 287, 288, 292,
                   293, 295, 296, 297, 298, 299, 301, 305, 307, 308, 309, 318, 319, 322, 323, 324, 328, 330, 332, 333,
                   334, 335, 336, 338, 339, 340, 342, 344, 347,
                   349, 350, 353, 356, 359, 363, 365, 366, 368, 369, 371, 374, 375, 376, 377, 379, 384, 385, 387, 389,
                   390, 395, 396, 397, 398, 400, 403, 404, 406, 413,
                   414, 417, 418, 419, 427, 428, 435, 437, 438, 439, 442, 444, 445, 447, 449, 452, 455, 457, 460, 466,
                   468, 471, 473, 474, 475, 476, 477, 478, 479,
                   484, 486, 488, 491, 492, 493, 495, 496, 500, 501, 504, 505, 507, 515, 516, 518, 520, 523, 526, 527,
                   539, 542, 547, 551, 554, 558, 563, 564, 566,
                   567, 568, 569, 570, 571, 574, 576, 580, 583, 586, 587, 588, 593, 595, 598, 599, 601, 606, 610, 611,
                   613, 616, 621, 622, 623, 626, 631, 637, 638,
                   639, 640, 641, 642, 645, 650, 652, 655, 665, 667, 668, 670, 674, 676, 678, 679, 700, 702, 703, 708,
                   714, 715, 716, 718, 720, 728, 732, 734, 738,
                   741, 743, 744, 747, 750, 754, 755, 770, 771, 772, 773, 780, 785, 787, 788, 789, 791, 793, 797, 800,
                   802]
    num_images = correct_img.copy()
    absent = [52, 119, 136, 256, 312, 315, 424, 485, 494, 572, 615, 633, 647, 691, 711, 792, 801]
    for i in range(len(correct_img)):
        for a in absent:
            if num_images[i] > a:
                num_images[i] = num_images[i] - 1

    for image_file in os.listdir(images_dir):
        if int(image_file[-7:-4]) in correct_img:
            image_path = os.path.join(images_dir, image_file)
            image = Image.open(image_path)
            images.append(image)
    images = np.stack(images)
    return images, num_images


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
# filter_bank = gabor_bank()

# get  images
train_images, train_nums = get_train_images(train_images_dir)
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

no_train_imgs = train_images.shape[0]
print(train_landmarks)

# define a list to store the features for all the training images
training_features = []
all_positive = []
all_negative = []
# iterate over the training images
for i in range(len(train_nums)):
    # get each image
    face_img = train_images[i]

    # get the corresponding landmarks
    print(train_nums[i])
    landmarks = np.array(train_landmarks)[train_nums[i], 2:]


    # create an array to store the landmarks for further processing - p_landmarks
    p_landmarks = np.zeros((19, 2), dtype=np.int)
    j = 0
    for l_i in range(19):
        p_landmarks[l_i] = (landmarks[j], landmarks[j + 1])
        j = j + 2

    # print the landmarks for the ith image (to check them)
    print("p_land", p_landmarks)

    # convert to grayscale
    bw_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("orig_img ", bw_img)
    # cv2.waitKey()

    ####################################  Training examples generation #####################################################
    p_examples = np.zeros((19, 9, 49, 13, 13)).astype('float32')  # positive patches 13x13
    n_examples = np.zeros((19, 16, 49, 13, 13)).astype('float32')  # negative patches 13x13
    rois = np.zeros((19, 37, 37))  # 37x37 roi around the ground truth feature point
    bw_copy = bw_img.copy()

    for l in range(len(p_landmarks)):
        # generate the examples based on the ground truth feature points
        # bw_copy=cv2.circle(bw_copy,p_landmarks[l], radius=0, color=255, thickness=5)
        print(i, ",", l)
        try:
            p_examples[l], n_examples[l], rois[l] = get_training_examples(bw_img, p_landmarks[l])
        except:
            pass
    all_positive.append(p_examples)
    all_negative.append(n_examples)


#################################### Training the regressor ###################################

for landmark in range(19):
    train_data = []
    train_label = []
    # AdaBoostRegressor
    # Choosing Decision Tree as the weak learner
    DT = DecisionTreeClassifier(max_depth=1)
    boostingClassifier = AdaBoostClassifier(n_estimators=50, base_estimator=DT, learning_rate=1)
    # Printing all the parameters of Adaboost
    print(boostingClassifier)

    for im in range(len(all_positive)):
        for p in range(9):
            vector = np.reshape(all_positive[im][landmark][p], (np.size(all_positive[im][landmark][p])))
            train_data.append(vector)
            train_label.append(1)
        for n in range(16):
            vector = np.reshape(all_negative[im][landmark][n], (np.size(all_negative[im][landmark][n])))
            train_data.append(vector)
            train_label.append(0)
    train_label_df = pd.DataFrame(train_label)
    train_data_df = pd.DataFrame(train_data)

    # Creating the model on Training Data - we might need to restructure the input data
    train_label_df = np.ravel(train_label_df)
    model_trained = boostingClassifier.fit(train_data_df, train_label_df)

    # save the model's weights
    os.makedirs(args.model_save_dir, exist_ok=True)
    # model_save_path = os.path.join(args.model_save_dir, f"boosting_{args.dataset}.model")
    model_save_path = os.path.join(args.model_save_dir, f"landmark {landmark}_{args.dataset}.model")
    save_model(boostingClassifier, model_save_path)
    print("The model is saved to:", model_save_path)

print("Training is completed!")

#################################### Training the regressor ###################################


# AdaBoostRegressor
# Choosing Decision Tree as the weak learner
DT = DecisionTreeClassifier(max_depth=1)
boostingClassifier = AdaBoostClassifier(n_estimators=50, base_estimator=DT, learning_rate=1)

# Printing all the parameters of Adaboost
print(boostingClassifier)

# Creating the model on Training Data - we might need to restructure the input data
train_label_df = np.ravel(train_label_df)
model_trained = boostingClassifier.fit(train_data_df, train_label_df)

# save the model's weights
os.makedirs(args.model_save_dir, exist_ok=True)
#model_save_path = os.path.join(args.model_save_dir, f"boosting_{args.dataset}.model")
model_save_path = os.path.join(args.model_save_dir, f"with_gabor_{args.dataset}.model")
save_model(boostingClassifier, model_save_path)
print("The model is saved to:", model_save_path)
print("Training is completed!")

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
