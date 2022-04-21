import os
import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import chirp, find_peaks, peak_widths
from FaceDetection import *
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
    #plt.plot(np.array(range(0, np.size(vert_diff_summed))), vert_diff_summed)
    plt.title("Vertical Histogram of the Eye")
    plt.xlabel("Rows")
    plt.ylabel("Pixel Intesity Differences Summed Over Columns")
    #plt.show()
    #plt.close()

    # then find x
    diff_hor = np.zeros((np.shape(image[:, 0])[0], np.shape(image[0])[0] - 1))
    for i in range(np.shape(image[0])[0] - 1):
        diff_hor[:, i] = np.abs(image_signed[:, i + 1] - image_signed[:, i])
    hor_diff_summed = np.sum(diff_hor, axis=1)
    max_hor = np.where(hor_diff_summed == np.max(hor_diff_summed))
    y = max_hor[0][0]
    #plt.plot(np.array(range(0, np.size(hor_diff_summed))), hor_diff_summed)
    plt.title("Horizontal Histogram of the Eye")
    plt.xlabel("Columns")
    plt.ylabel("Pixel Intesity Differences Summed Over Rows")
    #plt.show()
    #plt.close()
    return y, x

def rotate_face(img,left_pupil,right_pupil):
    # rotate the face to align the pupils to be on the same level
    degree = np.arctan((right_pupil[0]-left_pupil[0])/(right_pupil[1]-left_pupil[1]))*180/np.pi
    (h, w) = img.shape
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), degree, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def find_roi (image, eye_distance, left_eye_loc, right_eye_loc,mouth_loc):  # eye distance is integer, rest are tuples (y,x)
    rows, cols, = np.shape(image)  # size of image
    # center locations of the features
    A_loc = (int(left_eye_loc[0]), int(left_eye_loc[1]*2/3))  # at the same row with the left eye, 2/3 distance from the beggining of the image
    A1_loc = (int(right_eye_loc[0]), int((2*right_eye_loc[1]+cols)/3))  # right eye, " " " ,2/3 distance from the end
    B_loc = (int(left_eye_loc[0]), int(left_eye_loc[1]+eye_distance/4))  #left eye, same row, 1/4*ED from left eye(mid point between nose end pupil)
    B1_loc = (int(right_eye_loc[0]), int(right_eye_loc[1]-eye_distance/4))  # right eye
    D_loc = (int(left_eye_loc[0]*3/4), int((right_eye_loc[1]+2*left_eye_loc[1])/3))  # left eyebrow inner side
    D1_loc = (int(right_eye_loc[0]*3/4), int((2*right_eye_loc[1]+left_eye_loc[1])/3))  # right eyebrow inner side
    E_loc = (int(D_loc[0]), int((3*left_eye_loc[1])/5))  # left eyebrow outer side
    E1_loc = (int(D_loc[0]), int((2*cols+3*right_eye_loc[1])/5))  # right eyebrow outer side
    F_loc = (int((left_eye_loc[0]+D_loc[0])/2), left_eye_loc[1])  # left eye upper
    F1_loc = (int((right_eye_loc[0]+D1_loc[0])/2), right_eye_loc[1])  # right eye upper
    G_loc = (int((3*left_eye_loc[0]-D_loc[0])/2), left_eye_loc[1])  # left eye lower
    G1_loc = (int((3*right_eye_loc[0]-D1_loc[0])/2), right_eye_loc[1])  # right eye lower
    H_loc = (int((3 * left_eye_loc[0] + 4 * mouth_loc[0]) / 7), B_loc[1])  # nose left side
    H1_loc = (int((3 * right_eye_loc[0] + 4 * mouth_loc[0]) / 7), B1_loc[1])  # nose left side
    I_loc = (mouth_loc[0],int((mouth_loc[1]+4*left_eye_loc[1])/5))  # mouth left
    J_loc = (mouth_loc[0], int((mouth_loc[1] + 4 * right_eye_loc[1]) / 5))  # mouth right
    N_loc = (int((left_eye_loc[0]+right_eye_loc[0]+5*(H_loc[0]+H1_loc[0]))/12), int((H_loc[1]+H1_loc[1])/2))  # nose middle point
    K_loc = (int((N_loc[0]+4*mouth_loc[0])/5), mouth_loc[1])
    L_loc = (int((6*mouth_loc[0]-N_loc[0])/5), mouth_loc[1])
    M_loc = (rows, mouth_loc[1])

    locs = [A_loc,A1_loc,B_loc,B1_loc,D_loc,D1_loc,E_loc,E1_loc,F_loc,F1_loc,G_loc,G1_loc,H_loc,H1_loc,I_loc,J_loc,N_loc,K_loc,L_loc,M_loc]
    for loc_tuple in locs:
        cv2.circle(image, (loc_tuple[1], loc_tuple[0]), radius=0, color=0, thickness=5)
    cv2.imshow("é",image)
    cv2.waitKey(0)
    print("hi")


# path to image in string, then upload to matrix
img_path = os.path.abspath(os.path.join(os.getcwd(), "..", "CS484-555_Project", "images", "face3_orig.jpg")).replace('\\', '/')
img = cv2.imread(img_path)

#cv2.imshow("image", img)
#cv2.waitKey(0)

# detect the faces
face_img = detect_face(imagePath="image.png")
cv2.imshow("face", face_img)
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

bw_copy = bw_img.copy()
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

copy_left = upper_left.copy()
copy_right = upper_right.copy()

y_left, x_left = find_pupil(upper_left)
y_right, x_right = find_pupil(upper_right)
print(y_left, ", ", x_left)
print(y_right, ", ", x_right)

# the position of right pupil in the original image
x_right_real = x_right + np.size(upper_left[0])

# align the face so that the eye axis is parallel to horizon, if it is nearly parallel don't
if abs(y_left-y_right) > 1:
    aligned = rotate_face(bw_img, (y_left, x_left), (y_right, x_right_real))
    upper_face = aligned[0:int(np.ceil(rows/2)), :]
    upper_left = upper_face[:, 0:int(np.ceil(cols/2))]
    upper_right = upper_face[:, int(np.ceil(cols/2)):]
    y_left, x_left = find_pupil(upper_left)
    y_right, x_right = find_pupil(upper_right)
    print(y_left, ", ", x_left)
    print(y_right, ", ", x_right)
    cv2.imshow("aligned", aligned)
    bw_img = aligned

cv2.circle(copy_left, (x_left, y_left), radius=0, color=(0, 0, 255), thickness=5)
cv2.circle(copy_right, (x_right, y_right), radius=0, color=(0, 255, 0), thickness=5)
cv2.circle(face_img, (x_left, y_left), radius=0, color=(0, 0, 255), thickness=5)
cv2.circle(face_img, (x_right_real, y_right), radius=0, color=(0, 0, 255), thickness=5)

cv2.imshow("eye l",copy_left)
cv2.imshow("eye r",copy_right)
cv2.imshow("eyes",bw_img)


ED = x_right_real-x_left  # eye distance

mouth_region = bw_img[y_left+int(0.85*ED): y_left+int(1.5*ED), x_left: x_right_real]
cv2.imshow("mouth region", mouth_region)
cv2.waitKey(0)

# find edge and threshold the mouth region
diff_ver = np.zeros((np.shape(mouth_region[:, 0])[0] - 1, np.shape(mouth_region[0])[0]))
image_signed = np.array(mouth_region, dtype=np.int8)
for i in range(np.shape(mouth_region[:, 0])[0] - 1):
    diff_ver[i] = np.abs(image_signed[i + 1].reshape((1, np.size(mouth_region[i + 1]))) -
                         image_signed[i].reshape((1, np.size(mouth_region[i]))))

edge = diff_ver.copy()
positions = np.where(diff_ver <= np.max(diff_ver)*0.6)
edge[diff_ver < np.max(diff_ver)*0.2]=0
edge[diff_ver > np.max(diff_ver)*0.2]=255

#cv2.imwrite("mouth edges.png", edge)
cv2.imshow("mouth edges", edge)
cv2.waitKey(0)

vert_diff_summed = np.sum(diff_ver, axis=1)

peaks, _ = find_peaks(vert_diff_summed)
results_half = peak_widths(vert_diff_summed, peaks, rel_height=0.5)
plt.plot(np.array(range(0, np.size(vert_diff_summed))), vert_diff_summed)
plt.hlines(*results_half[1:], color="C2")
plt.title("Vertical Histogram of the Mouth")
plt.xlabel("Rows")
plt.ylabel("Pixel Intesity Differences Summed Over Columns")

plt.show()
plt.close()

widest_peak = peaks[np.where(results_half[0] == np.max(results_half[0]))][0]
mouth_y, mouth_x = int(widest_peak+y_left+0.85*ED), int((x_right_real+x_left)/2)
cv2.circle(face_img, (mouth_x, mouth_y), radius=0, color=(255, 0, 0), thickness=5)
cv2.imshow("mouth ", face_img)
cv2.waitKey(0)

find_roi(bw_img, ED, (y_left, x_left-10), (y_right, x_right_real-10),(mouth_y, mouth_x-10))
#cv2.imwrite("face 3 detected.png",face_img)

#mouthlen_div3 = int((x_right_real-x_left)/5)
#bw_copy = cv2.rectangle(bw_copy, (x_left-2*mouthlen_div3,int((y_left+0.85*ED+mouth_y)/2) ), (x_left+mouthlen_div3,int((y_left+1.5*ED+mouth_y)/2)), 255, 3)
#bw_copy = cv2.rectangle(bw_copy, (x_right_real-2*mouthlen_div3,int((y_left+0.85*ED+mouth_y)/2) ), (x_right_real+mouthlen_div3,int((y_left+1.5*ED+mouth_y)/2)), 255, 3)
#bw_copy = cv2.rectangle(bw_copy, (x_left-10,y_left-20 ), (int((x_right_real+x_left)/2)-10,y_left+20), 255, 3)

#bw_copy = cv2.rectangle(bw_copy, (int(0.85*ED),int(x_left-2*mouthlen_div3)), (int(1.5*ED+mouth_y),int(mouth_x-mouthlen_div3)), 255, 3)
#cv2.imshow("mouth rct ", bw_copy)
#cv2.waitKey(0)

#cv2.imwrite("roi detected.png",bw_copy)