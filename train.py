import argparse
import os
import random
from random import sample

from PIL import Image
import numpy as np
import pandas as pd

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

import pickle


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


def get_training_examples(image, landmark):
    x, y = landmark
    half_roi_size = 6
    gt = image[x - half_roi_size:x + half_roi_size, y - half_roi_size:y + half_roi_size]
    x1 = x + 1
    x2 = x - 1
    y1 = y + 1
    y2 = y - 1
    p1 = image[x1 - half_roi_size:x1 + half_roi_size, y - half_roi_size:y + half_roi_size]
    p2 = image[x2 - half_roi_size:x2 + half_roi_size, y - half_roi_size:y + half_roi_size]
    p3 = image[x - half_roi_size:x + half_roi_size, y1 - half_roi_size:y1 + half_roi_size]
    p4 = image[x - half_roi_size:x + half_roi_size, y2 - half_roi_size:y2 + half_roi_size]
    p5 = image[x1 - half_roi_size:x1 + half_roi_size, y1 - half_roi_size:y1 + half_roi_size]
    p6 = image[x2 - half_roi_size:x2 + half_roi_size, y2 - half_roi_size:y2 + half_roi_size]
    p7 = image[x1 - half_roi_size:x1 + half_roi_size, y2 - half_roi_size:y2 + half_roi_size]
    p8 = image[x2 - half_roi_size:x2 + half_roi_size, y1 - half_roi_size:y1 + half_roi_size]
    print(x,y)
    print(image.shape)
    print(x1 - half_roi_size)
    print(x1 + half_roi_size)
    print(y1 - half_roi_size)
    print(y1 + half_roi_size)




    # positive examples
    p_eg = np.array([gt, p1, p2, p3, p4, p5, p6, p7, p8])
    print("p_eg", p_eg)
    print("p_eg", p_eg.shape)
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

    # true roi
    true_roi = image[x - 18:x + 18, y - 18:y + 18]

    return p_eg, n_eg, true_roi



def get_patch(image, landmark):
    # get landmark coordinates
    x, y = landmark

    # get the 13x13 patch around the landmark
    patch = image[x - 6:x + 6, y - 6:y + 6]

    return patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="lfpw", help='Name of the dataset.')
    parser.add_argument('--dataset_dir', type=str, default="./datasets/lfpw/processed",
                        help='Directory of the dataset that the regressor is trained on.')
    parser.add_argument('--model_save_dir', type=str, default="./saved_models",
                        help='Where the trained model will be saved.')
    args = parser.parse_args()

    train_data_dir = os.path.join(args.dataset_dir, "train")
    images_dir = os.path.join(train_data_dir, "images")
    landmarks_path = os.path.join(train_data_dir, "landmarks.csv")

    images = get_images(images_dir)
    df_landmarks = pd.read_csv(landmarks_path, index_col=0)

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
