import os
import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import chirp, find_peaks, peak_widths

def find_pupil(image):
    # first find y
    diff_ver = np.zeros((np.shape(image[:, 0])[0] - 1, np.shape(image[0])[0]))
    image_signed = np.array(image, dtype=np.int8)
    for i in range(np.shape(image[:, 0])[0] - 1):
        diff_ver[i] = np.abs(image_signed[i + 1].reshape((1, np.size(image[i + 1]))) -
                             image_signed[i].reshape((1, np.size(image[i]))))
    vert_diff_summed = np.sum(diff_ver, axis=0)
    max_vert = np.where(vert_diff_summed == np.max(vert_diff_summed))
    x = max_vert[0][0] # it is in array format even though a single element, y is the integer we need

    # then find x
    diff_hor = np.zeros((np.shape(image[:, 0])[0], np.shape(image[0])[0] - 1))
    for i in range(np.shape(image[0])[0] - 1):
        diff_hor[:, i] = np.abs(image_signed[:, i + 1] - image_signed[:, i])
    hor_diff_left_summed = np.sum(diff_hor, axis=1)
    max_hor = np.where(hor_diff_left_summed == np.max(hor_diff_left_summed))
    y = max_hor[0][0]
    return y, x

def rotate_face(img,left_pupil,right_pupil):
    # rotate the face to align the pupils to be on the same level
    degree = np.arctan((right_pupil[0]-left_pupil[0])/(right_pupil[1]-left_pupil[1]))*180/np.pi
    (h, w) = img.shape
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), degree, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated


# path to image in string, then upload to matrix
img_path = os.path.abspath(os.path.join(os.getcwd(), "..", "CS484-555_Project", "images", "face3.jpg")).replace('\\', '/')
face_img = cv2.imread(img_path)
cv2.imshow("face 1", face_img)
cv2.waitKey(0)

#percent by which the image is resized
scale_percent = 200

#calculate the 50 percent of original dimensions
width = int(face_img.shape[1] * scale_percent / 100)
height = int(face_img.shape[0] * scale_percent / 100)

# dsize
dsize = (width, height)

# resize image
face_img = cv2.resize(face_img, dsize)

# convert to grayscale
bw_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
rows, cols = np.shape(bw_img)

# divide the picture into upper and lower halves
upper_face = bw_img[0:int(np.ceil(rows/2)), :]
lower_face = bw_img[int(np.ceil(rows/2)):, :]
cv2.imshow("upper", upper_face)
cv2.imshow("lower", lower_face)
cv2.waitKey(0)

# divide the face into left and right (our left and right, not theirs)
upper_left = upper_face[:, 0:int(np.ceil(cols/2))]
upper_right = upper_face[:, int(np.ceil(cols/2)):]
cv2.imshow("upper left", upper_left)
cv2.imshow("upper right", upper_right)
cv2.waitKey(0)

y_pad = np.zeros((np.shape(upper_left[:, 0])[0], 1))
x_pad = np.zeros((1, np.shape(upper_left[0, :])[0]))

copy_left = upper_left.copy()
copy_right = upper_right.copy()

y_left, x_left = find_pupil(upper_left)
y_right, x_right = find_pupil(upper_right)
print(y_left,", ",x_left)
print(y_right,", ",x_right)

# the position of right pupil in the original image
x_right_real = x_right + np.size(upper_left[0])

# align the face so that the eye axis is parallel to horizon, if it is nearly parallel don't
if abs(y_left-y_right)>1:
    aligned = rotate_face(bw_img,(y_left,x_left),(y_right,x_right_real))
    upper_face = aligned[0:int(np.ceil(rows/2)), :]
    upper_left = upper_face[:, 0:int(np.ceil(cols/2))]
    upper_right = upper_face[:, int(np.ceil(cols/2)):]
    y_left, x_left = find_pupil(upper_left)
    y_right, x_right = find_pupil(upper_right)
    print(y_left,", ",x_left)
    print(y_right,", ",x_right)
    cv2.imshow("aligned",aligned)
    bw_img=aligned

cv2.circle(copy_left, (x_left, y_left), radius=0, color=(0, 0, 255), thickness=5)
cv2.circle(copy_right, (x_right, y_right), radius=0, color=(0, 255, 0), thickness=5)
cv2.circle(bw_img, (x_left, y_left), radius=0, color=(255, 0, 0), thickness=5)
cv2.circle(bw_img, (x_right_real, y_right), radius=0, color=(255, 255, 0), thickness=5)

cv2.imshow("eye l",copy_left)
cv2.imshow("eye r",copy_right)
cv2.imshow("eyes",bw_img)


ED = x_right_real-x_left  # eye distance

mouth_region = bw_img[y_left+int(0.85*ED) : y_left+int(1.5*ED), x_left : x_right_real]
cv2.imshow("mouth region", mouth_region)
cv2.waitKey(0)

# find edge and threshold the mouth region
diff_ver = np.zeros((np.shape(mouth_region[:, 0])[0] - 1, np.shape(mouth_region[0])[0]))
image_signed = np.array(mouth_region, dtype=np.int8)
for i in range(np.shape(mouth_region[:, 0])[0] - 1):
    diff_ver[i] = np.abs(image_signed[i + 1].reshape((1, np.size(mouth_region[i + 1]))) -
                         image_signed[i].reshape((1, np.size(mouth_region[i]))))

edge = diff_ver.copy()
cv2.imshow("mouth edges", diff_ver)
cv2.waitKey(0)
positions = np.where(diff_ver <= np.max(diff_ver)*0.6)
edge[diff_ver < np.max(diff_ver)*0.15]=0
edge[diff_ver > np.max(diff_ver)*0.15]=255

cv2.imshow("mouth edges2", edge)
cv2.waitKey(0)

vert_diff_summed = np.sum(diff_ver, axis=0)

max_ver = np.where(vert_diff_summed == np.max(vert_diff_summed))
mouth_y, mouth_x = int(max_ver[0][0]+y_left+0.85*ED), int((x_right_real+x_left)/2)
cv2.circle(bw_img, (mouth_x, mouth_y), radius=0, color=255, thickness=5)

cv2.imshow("mouth ",bw_img)
cv2.waitKey(0)

peaks, _ = find_peaks(vert_diff_summed)
results_half = peak_widths(vert_diff_summed, peaks, rel_height=0.5)
plt.plot(np.array(range(0,np.size(vert_diff_summed))), vert_diff_summed)
plt.hlines(*results_half[1:], color="C2")
plt.show()
plt.close()
