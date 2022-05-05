from math import pi, sqrt, sin, cos, ceil, floor
from ExtractROIs import *
import numpy as np
import cv2


def gabor_bank(size=13, no_orientation=8, no_freq=6):
    """

    :param size: the size of the gabor filters (size-by-size)
    :param no_orientation: the number of different orientations that filters will have
    :param no_freq: the number of different frequencies that filters will have
    :return: a bank of 48 Gabor filters at 8 orientations and 6 spatial frequencies (2:12 pixels/cycle at 1⁄2 octave steps).
    """
    # define the Gabor parameters
    gama = 1 / sqrt(pi * 0.9025)
    eta = 1 / sqrt(pi * 0.58695)
    f_max = 0.5

    # create a list for the filter bank
    filter_bank = []
    for o in range(no_orientation):  # iterate over the orientations
        for f in range(no_freq):  # iterate over the frequencies(scale)

            # generate the frequencies
            f0 = f_max / ((sqrt(2)) ** f)

            # generate the orientations
            theta = (o/no_orientation) * pi

            # define the filter's size
            x = np.repeat(np.asarray(range(-floor(size / 2), ceil(size / 2))).reshape(1, size), size, axis=0)
            y = np.transpose(x)

            # compute the filter based on the orientation parameter theta
            x_p = cos(theta) * x + sin(theta) * y
            y_p = -sin(theta) * x + cos(theta) * y

            # create the normalized filter - in the spatial domain
            h = f0 ** 2 / pi / gama / eta * np.multiply(
                np.exp(-f0 ** 2 / gama ** 2 * x_p ** 2 - f0 ** 2 / eta ** 2 * y_p ** 2),
                np.exp(2j * pi * f0 * x_p))

            h2 = cv2.getGaborKernel(ksize=(size,size),sigma=gama/f0/sqrt(2),theta=theta,lambd=f0,gamma=gama/eta)
            # check the filters
            """"""

            # resize the filter to display it in more details
            scale_percent = 500  # define the desired scale
            width = int(h.shape[1] * scale_percent / 100)
            height = int(h.shape[0] * scale_percent / 100)

            # normalize the filter values to 256 grayscale levels
            c_h = cv2.resize(np.real(h * 255 / np.max(np.real(h - np.min(np.abs(h))))), (width, height))

            # display the filter
            #cv2.imshow("h", c_h)
            #cv2.waitKey()

            # store the current filter in the filter bank
            filter_bank.append(h)

    return filter_bank


def feature_extraction(filters_bank, ROIs):
    """
    :param filters_bank: a list consisting of gabor filters (in np array form), for this project the length is generally 48
    :param ROIs: a list of images ROIs (in np array form), for this project the length is generally 20
    :return:  in the result, each row belongs to one feature, each of them are 49 linearized images first is original, rest is with filters
    """

    # define the filter's parameters
    no_filters = len(filters_bank)
    no_ROI = len(ROIs)
    roi_size = np.shape(ROIs[0])[0]
    # define a numpy array to store the vectors of features
    feature_vectors = np.zeros((no_ROI, (no_filters + 1) * roi_size * roi_size))

    # create a list for the 2D filtered ROIs
    filtered_ROIs_2D = []
    for idx_roi in range(no_ROI):  # iterate over the ROIs

        # extract the current ROI
        roi = ROIs[idx_roi]

        # create a list to store the current filtered ROI for displaying
        filtered_roi_2D = []

        # store the unfiltered(original) gray scale values of the current ROI as a 2D array
        filtered_roi_2D.append(roi)

        # store the unfiltered(original) gray scale values of the current ROI as a 1D vector
        orig_vect = np.reshape(roi, (1, np.size(roi)))

        # store the vector of the ROI's original gray scale values in the feature_vectors array
        feature_vectors[idx_roi, 0:roi_size ** 2] = orig_vect

        # iterate over the filters
        for f in range(no_filters):

            # filter the current ROI using the Gabor filters
            f_roi = cv2.filter2D(src=roi.astype('float64'), ddepth=-1, kernel=np.abs(filters_bank[f]))

            # store all the filtered ROI's features as a 2D arrays in the filtered_roi list
            filtered_roi_2D.append(f_roi)

            # store all the filtered ROI's features as a 1D array in the feature_vectors array
            feature_vectors[idx_roi, (f + 1) * roi_size ** 2:(f + 2) * roi_size ** 2] = np.reshape(f_roi, (1, np.size(f_roi)))

            # check the filtered ROI

            # resize the filter to display it in more details
            scale_percent = 500  # define the desired scale
            width = int(f_roi.shape[1] * scale_percent / 100)
            height = int(f_roi.shape[0] * scale_percent / 100)

            # resize the filtered ROI
            f_roi = cv2.resize(f_roi / np.max(f_roi) * 255, (width, height))

            # display the filtered ROI
            #cv2.imshow("ROI:" + str(idx_roi) + ", filter:" + str(f), f_roi.astype('uint8'))
            #cv2.waitKey()

            # print the filtering progress
            print("Filtering ROI " + str(idx_roi) + " with filter " + str(f))

        filtered_ROIs_2D.append(filtered_roi_2D)

    return feature_vectors,filtered_ROIs_2D

def feature_extraction1(filters_bank, ROI):
    """
    :param filters_bank: a list consisting of gabor filters (in np array form), for this project the length is
                generally 48
    :param ROIs: a list of images ROIs (in np array form), for this project the length is generally 20
    :return:  in the result, each row belongs to one feature, each of them are 49 linearized images first is original,
                rest is with filters
    """

    # define the filter's parameters
    no_filters = len(filters_bank)

    # create a list for the 2D filtered ROIs
    filtered_roi_2D = []

    # store the unfiltered(original) gray scale values of the current ROI as a 2D array
    filtered_roi_2D.append(ROI)

    # iterate over the filters
    for f in range(no_filters):
        # filter the current ROI using the Gabor filters
        f_roi = cv2.filter2D(src=ROI.astype('float64'), ddepth=-1, kernel=np.abs(filters_bank[f]))
        f_roi=f_roi.astype('float32')
        # store all the filtered ROI's features as a 2D arrays in the filtered_roi list
        filtered_roi_2D.append(f_roi)

        # check the filtered ROI
        # resize the filter to display it in more details
        scale_percent = 500  # define the desired scale
        width = int(f_roi.shape[1] * scale_percent / 100)
        height = int(f_roi.shape[0] * scale_percent / 100)

        # resize the filtered ROI
        f_roi = cv2.resize(f_roi / np.max(f_roi) * 255, (width, height))

        # display the filtered ROI
        #cv2.imshow("Filter:" + str(f), f_roi.astype('uint8'))
        #cv2.waitKey()

    return filtered_roi_2D



