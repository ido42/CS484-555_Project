import argparse
import os
import random
from random import sample

from PIL import Image
import numpy as np
import pandas as pd

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import cv2
import pickle
from FeatureExtraction import *
fb = gabor_bank()


def get_images(images_dir):
    print("Reading the images from the dataset.")
    images = []
    for image_file in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_file)
        image = Image.open(image_path)
        images.append(image)
    images = np.stack(images)
    return images

def get_landmarks(landmarks_dir):
    df_landmarks=[]
    for landmarks_file in os.listdir(landmarks_dir):
        landmark_path = os.path.join(landmarks_dir, landmarks_file)
        landmarks = np.loadtxt(landmark_path)
        landmarks = landmarks.reshape(-1, 2)[:, :2]
        df_landmarks.append(landmarks)

    df_landmarks = np.stack(df_landmarks)
    return df_landmarks

def save_model(model, model_save_path):
    with open(model_save_path, 'wb') as model_save_file:
        pickle.dump(model, model_save_file)


def positive_ex(filtered_ROIs):
    l = [12, 13, 14]
    positive_ex = []
    filt_num=len(filtered_ROIs)

    for x in l:
        for y in l:
            filtered_ex=np.zeros((filt_num,13,13))
            for f in range(filt_num):
                filtered_ex[f, :, :] = filtered_ROIs[f][y:y + 13, x:x + 13]
            positive_ex.append(filtered_ex)
    return positive_ex


def negative_ex(filtered_ROIs):
    n_eg = []
    x = 19
    y = 19
    half_roi_size = 6
    random_locs=[]
    filt_num = len(filtered_ROIs)
    while len(random_locs) != 8:
        x_err=random.randint(-2, 2)
        y_err=random.randint(-2, 2)
        if not(abs(x_err) == 1 and abs(y_err) == 1):
            random_locs.append((x+x_err, y+y_err))
    while len(random_locs) != 16:
        x_rand = random.randint(-10,10)
        y_rand = random.randint(-10,10)
        if not(abs(x_rand) <= 2 and abs(y_rand) <= 2):
            random_locs.append((x+x_rand, y+y_rand))

    for l in random_locs:
        filtered_ex = np.zeros((filt_num, 13, 13))
        for i in range(len(filtered_ROIs)):
            f=filtered_ROIs[i]
            x, y = l
            filtered_ex[i] = f[x-half_roi_size:x + half_roi_size+1, y-half_roi_size:y+half_roi_size+1]

        n_eg.append(filtered_ex)
    return n_eg

def get_training_examples(image, landmark, filter_bank = fb):
    # generate the Gabor filters


    # true roi
    x, y = landmark
    true_roi = image[x - 18:x + 19, y - 18:y + 19]
    pad_x = 37 - np.shape(true_roi)[1]
    pad_y = 37 - np.shape(true_roi)[0]
    true_roi=np.pad(true_roi, ((0, pad_y), (0, pad_x)), 'constant')


    # get the features using the ground truth rois
    features_2D = feature_extraction1(filter_bank, true_roi)
    #all_filtered_patches = filtered_patch(features_2D)
    p_eg=positive_ex(features_2D)
    n_eg=negative_ex(features_2D)
    """
    half_roi_size = 6
    gt = image[x - half_roi_size:x + half_roi_size+1, y - half_roi_size:y + half_roi_size+1]
    x1 = x + 1
    x2 = x - 1
    y1 = y + 1
    y2 = y - 1
    p1 = image[x1 - half_roi_size:x1 + half_roi_size + 1, y - half_roi_size:y + half_roi_size + 1]
    p2 = image[x2 - half_roi_size:x2 + half_roi_size+1, y - half_roi_size:y + half_roi_size+1]
    p3 = image[x - half_roi_size:x + half_roi_size+1, y1 - half_roi_size:y1 + half_roi_size+1]
    p4 = image[x - half_roi_size:x + half_roi_size+1, y2 - half_roi_size:y2 + half_roi_size+1]
    p5 = image[x1 - half_roi_size:x1 + half_roi_size+1, y1 - half_roi_size:y1 + half_roi_size+1]
    p6 = image[x2 - half_roi_size:x2 + half_roi_size+1, y2 - half_roi_size:y2 + half_roi_size+1]
    p7 = image[x1 - half_roi_size:x1 + half_roi_size+1, y2 - half_roi_size:y2 + half_roi_size+1]
    p8 = image[x2 - half_roi_size:x2 + half_roi_size+1, y1 - half_roi_size:y1 + half_roi_size+1]
    """
    """
    print(x,y)
    print(image.shape)
    print(x1 - half_roi_size)
    print(x1 + half_roi_size)
    print(y1 - half_roi_size)
    print(y1 + half_roi_size)
    """
    """
    # positive examples
    p_eg = np.array([gt, p1, p2, p3, p4, p5, p6, p7, p8])
    p_eg_padded_list=[]
    for reg in p_eg:
        pad_x = 13-np.shape(reg)[1]
        pad_y = 13-np.shape(reg)[0]
        p_eg_padded_list.append(np.pad(reg, ((0, 0), (pad_y, pad_x)), 'constant'))
    # print("p_eg", p_eg)
    # print("p_eg", p_eg.shape)
    p_eg_padded=np.asarray(p_eg_padded_list)

    n_eg=[]
    while len(n_eg) != 8:
        x_err=random.randint(-2,2)
        y_err=random.randint(-2,2)
        ex = image[x+x_err - half_roi_size:x+x_err + half_roi_size + 1, y+y_err - half_roi_size:y+y_err + half_roi_size + 1]
        pad_x = 13 - np.shape(ex)[1]
        pad_y = 13 - np.shape(ex)[0]
        ex = np.pad(ex, ((0, 0), (pad_y, pad_x)), 'constant')
        if not(abs(x_err) == 1 and abs(y_err) == 1):
            n_eg.append(ex)
    while len(n_eg)!=16:
        x_rand=random.randint(-15,15)
        y_rand=random.randint(-15,15)
        ex = image[x+x_rand - half_roi_size:x+x_rand + half_roi_size + 1, y+y_rand - half_roi_size:y+y_rand + half_roi_size + 1]
        pad_x = 13 - np.shape(ex)[1]
        pad_y = 13 - np.shape(ex)[0]
        ex = np.pad(ex, ((0, 0), (pad_y, pad_x)), 'constant')
        if not(abs(x_rand)<=2and abs(y_rand)<=2):
            n_eg.append(ex)
    n_eg_fin=np.asarray(n_eg)
    """
    """
    # the first set of negative examples 2 pixel apart from the ground truth
    x3 = x + 2
    x4 = x - 2
    y3 = y + 2
    y4 = y - 2
    n1 = image[x3 - half_roi_size:x3 + half_roi_size, y - half_roi_size:y + half_roi_size]
    n2 = image[x4 - half_roi_size:x4 + half_roi_size, y - half_roi_size:y + half_roi_size]
    n3 = image[x - half_roi_size:x + half_roi_size, y3 - half_roi_size:y3 + half_roi_size]
    n4 = image[x - half_roi_size:x + half_roi_size, y4 - half_roi_size:y4 + half_roi_size]
    n5 = image[x3 - half_roi_size:x3 + half_roi_size, y3 - half_roi_size:y3 + half_roi_size]
    n6 = image[x4 - half_roi_size:x4 + half_roi_size, y4 - half_roi_size:y4 + half_roi_size]
    n7 = image[x3 - half_roi_size:x3 + half_roi_size, y4 - half_roi_size:y4 + half_roi_size]
    n8 = image[x4 - half_roi_size:x4 + half_roi_size, y3 - half_roi_size:y3 + half_roi_size]

    # the second set of negative examples random n pixel apart from the ground truth
    x9, x10, x11, x12 = random.sample(range(x-18, x-4), 4)
    x13, x14, x15, x16 = random.sample(range(x+4, x+18), 4)
    y9, y10, y11, y12 = random.sample(range(x-18, x-4), 4)
    y13, y14, y15, y16 = random.sample(range(x+4, x+18), 4)
    n9 = image[x9 - half_roi_size:x9 + half_roi_size, y9 - half_roi_size:y9 + half_roi_size]
    n10 = image[x10 - half_roi_size:x10 + half_roi_size, y10 - half_roi_size:y10 + half_roi_size]
    n11 = image[x11 - half_roi_size:x11 + half_roi_size, y11 - half_roi_size:y11 + half_roi_size]
    n12 = image[x12 - half_roi_size:x12 + half_roi_size, y12 - half_roi_size:y12 + half_roi_size]
    n13 = image[x13 - half_roi_size:x13 + half_roi_size, y13 - half_roi_size:y13 + half_roi_size]
    n14 = image[x14 - half_roi_size:x14 + half_roi_size, y14 - half_roi_size:y14 + half_roi_size]
    n15 = image[x15 - half_roi_size:x15 + half_roi_size, y15 - half_roi_size:y15 + half_roi_size]
    n16 = image[x16 - half_roi_size:x16 + half_roi_size, y16 - half_roi_size:y16 + half_roi_size]
    """

    # negative examples
   # n_eg = np.array([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16])

    return p_eg, n_eg, true_roi  # n_eg


def filtered_patch(filtered_ROIs):
    l= [12,13,14]
    all_filtered_roi=[]
    for r in filtered_ROIs:
        one_roi_filtered=[]
        for f in r:
            for x in l:
                for y in l:
                    pos=f[y:y+13,x:x+13]
                    one_roi_filtered.append(pos)
            all_filtered_roi.append(one_roi_filtered)
    return  all_filtered_roi

def get_patch(image, landmark):
    # get landmark coordinates
    x, y = landmark

    # get the 13x13 patch around the landmark
    patch = image[x - 6:x + 6, y - 6:y + 6]

    return patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="CK", help='Name of the dataset.')
    parser.add_argument('--dataset_dir', type=str, default="",
                        help='Directory of the dataset that the regressor is trained on.')
    parser.add_argument('--model_save_dir', type=str, default="./saved_models",
                        help='Where the trained model will be saved.')
    args = parser.parse_args()

    train_data_dir = os.path.join(args.dataset_dir, "train")
    images_dir = os.path.join(train_data_dir, "CK_train_images")
    landmarks_path = os.path.join(train_data_dir, "CK_train_landmarks")

    images = get_images(images_dir)
    df_landmarks = get_landmarks(landmarks_path)



    print("Training has started!")
    print(df_landmarks)

    # AdaBoostRegressor
    # Choosing Decision Tree  as the weak learner
    DTR = DecisionTreeRegressor()
    boostingClassifier = AdaBoostRegressor(n_estimators=50, base_estimator=DTR, learning_rate=1)

    # Printing all the parameters of Adaboost
    print(boostingClassifier)

    # Creating the model on Training Data
    # model_train = boostingClassifier.fit(, df_landmarks)

    os.makedirs(args.model_save_dir, exist_ok=True)
    model_save_path = os.path.join(args.model_save_dir, f"boosting_{args.dataset}.model")
    save_model(boostingClassifier, model_save_path)
    print("The model is saved to:", model_save_path)
    print("Training is completed!")


if __name__ == "__main__":
    main()
