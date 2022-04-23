import os
import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import chirp, find_peaks, peak_widths
from FaceDetection import *
from math import floor, ceil

def find_pupil(image):
    # first find y
    diff_ver = np.zeros((np.shape(image[:, 0])[0] - 1, np.shape(image[0])[0]))
    image_signed = np.array(image, dtype=np.int8)
    for i in range(np.shape(image[:, 0])[0] - 1):
        diff_ver[i] = np.abs(image_signed[i + 1].reshape((1, np.size(image[i + 1]))) -
                             image_signed[i].reshape((1, np.size(image[i]))))
    vert_diff_summed = np.sum(diff_ver, axis=0)
    max_vert = np.where(vert_diff_summed == np.max(vert_diff_summed))
    x = max_vert[0][0]  # it is in array format even though a single element, y is the integer we need
    # plt.plot(np.array(range(0, np.size(vert_diff_summed))), vert_diff_summed)
    plt.title("Vertical Histogram of the Eye")
    plt.xlabel("Rows")
    plt.ylabel("Pixel Intesity Differences Summed Over Columns")
    # plt.show()
    # plt.close()

    # then find x
    diff_hor = np.zeros((np.shape(image[:, 0])[0], np.shape(image[0])[0] - 1))
    for i in range(np.shape(image[0])[0] - 1):
        diff_hor[:, i] = np.abs(image_signed[:, i + 1] - image_signed[:, i])
    hor_diff_summed = np.sum(diff_hor, axis=1)
    max_hor = np.where(hor_diff_summed == np.max(hor_diff_summed))
    y = max_hor[0][0]
    # plt.plot(np.array(range(0, np.size(hor_diff_summed))), hor_diff_summed)
    plt.title("Horizontal Histogram of the Eye")
    plt.xlabel("Columns")
    plt.ylabel("Pixel Intesity Differences Summed Over Rows")
    # plt.show()
    # plt.close()
    return y, x


def rotate_face(img, left_pupil, right_pupil):
    # rotate the face to align the pupils to be on the same level
    degree = np.arctan((right_pupil[0] - left_pupil[0]) / (right_pupil[1] - left_pupil[1])) * 180 / np.pi
    (h, w) = img.shape
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), degree, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated


def find_roi(image, eye_distance, left_eye_loc, right_eye_loc,
             mouth_loc):  # eye distance is integer, rest are tuples (y,x)
    rows, cols, = np.shape(image)  # size of image
    roi_size = 37
    # center locations of the features
    A_loc = (int(left_eye_loc[0]), int(
        left_eye_loc[1] * 2 / 3))  # at the same row with the left eye, 2/3 distance from the beggining of the image
    A1_loc = (
    int(right_eye_loc[0]), int((2 * right_eye_loc[1] + cols) / 3))  # right eye, " " " ,2/3 distance from the end
    B_loc = (int(left_eye_loc[0]), int(left_eye_loc[
                                           1] + eye_distance / 4))  # left eye, same row, 1/4*ED from left eye(mid point between nose end pupil)
    B1_loc = (int(right_eye_loc[0]), int(right_eye_loc[1] - eye_distance / 4))  # right eye
    D_loc = (int(left_eye_loc[0] * 3 / 4), int((right_eye_loc[1] + 2 * left_eye_loc[1]) / 3))  # left eyebrow inner side
    D1_loc = (
    int(right_eye_loc[0] * 3 / 4), int((2 * right_eye_loc[1] + left_eye_loc[1]) / 3))  # right eyebrow inner side
    E_loc = (int(D_loc[0]), int((3 * left_eye_loc[1]) / 5))  # left eyebrow outer side
    E1_loc = (int(D_loc[0]), int((2 * cols + 3 * right_eye_loc[1]) / 5))  # right eyebrow outer side
    F_loc = (int((left_eye_loc[0] + D_loc[0]) / 2), left_eye_loc[1])  # left eye upper
    F1_loc = (int((right_eye_loc[0] + D1_loc[0]) / 2), right_eye_loc[1])  # right eye upper
    G_loc = (int((3 * left_eye_loc[0] - D_loc[0]) / 2), left_eye_loc[1])  # left eye lower
    G1_loc = (int((3 * right_eye_loc[0] - D1_loc[0]) / 2), right_eye_loc[1])  # right eye lower
    H_loc = (int((3 * left_eye_loc[0] + 4 * mouth_loc[0]) / 7), B_loc[1])  # nose left side
    H1_loc = (int((3 * right_eye_loc[0] + 4 * mouth_loc[0]) / 7), B1_loc[1])  # nose left side
    I_loc = (mouth_loc[0], int((mouth_loc[1] + 4 * left_eye_loc[1]) / 5))  # mouth left
    J_loc = (mouth_loc[0], int((mouth_loc[1] + 4 * right_eye_loc[1]) / 5))  # mouth right
    N_loc = (int((left_eye_loc[0] + right_eye_loc[0] + 5 * (H_loc[0] + H1_loc[0])) / 12),
             int((H_loc[1] + H1_loc[1]) / 2))  # nose middle point
    K_loc = (int((N_loc[0] + 4 * mouth_loc[0]) / 5), mouth_loc[1])
    L_loc = (int((6 * mouth_loc[0] - N_loc[0]) / 5), mouth_loc[1])
    M_loc = (rows - ceil(roi_size/2), mouth_loc[1])

    locs = [A_loc, A1_loc, B_loc, B1_loc, D_loc, D1_loc, E_loc, E1_loc, F_loc, F1_loc, G_loc, G1_loc, H_loc, H1_loc,
            I_loc, J_loc, N_loc, K_loc, L_loc, M_loc]
    rois = []
    img = image.copy()
    i=0
    for loc_tuple in locs:
        roi = image[int(loc_tuple[0] - np.floor(roi_size / 2)):int(loc_tuple[0] + np.ceil(roi_size / 2)),
              int(loc_tuple[1] - np.floor(roi_size / 2)):int(loc_tuple[1] + np.ceil(roi_size / 2))]
        rois.append(roi)

        img = cv2.circle(img, (loc_tuple[1], loc_tuple[0]), radius=0, color=0, thickness=5)
        # percent by which the image is resized
        scale_percent = 500

        # calculate the 50 percent of original dimensions
        width = int(roi.shape[1] * scale_percent / 100)
        height = int(roi.shape[0] * scale_percent / 100)

        # dsize
        dsize = (width, height)
        i=i+1
        # resize image
      #  roi_resized = cv2.resize(roi, dsize)
     #   cv2.imshow(str(i),roi_resized)
    #cv2.imshow("marked", img)

    cv2.waitKey(0)

    return rois, locs
