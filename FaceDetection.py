import numpy as np
import cv2


def detect_face(imagePath, cascadeClassifier="haarcascade_frontalface_alt.xml", scaleFactor=1.1, minNeighbors=4,
                minSize=(60, 60)):
    """
    Detects the face in an image using a pretrained Haar feature-based cascade classifier for frontal faces
    :param imagePath: string - the path of the input image
    :param cascadeClassifier: string - the path to the XML file containing serialized Haar cascade detector of faces
    (Viola-Jones algorithm) in the OpenCV library
    :param scaleFactor: float - how much the image size is reduced at each image scale.
    :param minNeighbors: float - how many neighbors each candidate rectangle should have to retain it.
    :param minSize: tuple - minimum possible object size. Objects smaller than that are ignored.
    :return: 2D array - a part of the original image containing the face
    """
    # read input image
    original_image = cv2.imread(imagePath)

    # convert the input image to grayscale
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # load the OpenCV Viola-Jones classifier and create a cascade object for face detection
    faceCascade = cv2.CascadeClassifier(cascadeClassifier)

    # detect the face in the input image using the classifier
    detectedFace = faceCascade.detectMultiScale(image=grayscale_image, scaleFactor=scaleFactor, minNeighbors=minNeighbors,
                                         minSize=minSize)

    # extract the corners for the face
    column, row, width, height = detectedFace[0]

    # extract the face from the original image
    face = original_image[row:row + int(width*1.1), column:column + height]  # image is made longer to include the chin

    # save the detected face
    cv2.imwrite("detected_face.png", face)

    return face

"""
# define the image path
image_path = "image.png"

# read input image
original_image = cv2.imread(image_path)

# display the original image
cv2.imshow('Original image', original_image)
cv2.waitKey(0)

# detect the faces
detected_face = detect_face(imagePath=image_path)

# display the face image
cv2.imshow('Detected face', detected_face)
cv2.waitKey(0)
"""
