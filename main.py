from FaceDetection import *
from ExtractROIs import *
from FeatureExtraction import *

# path to image in string, then upload to matrix
img_path = os.path.abspath(os.path.join(os.getcwd(), "..", "CS484-555_Project", "images", "face3_orig.jpg")).replace(
    '\\', '/')
img = cv2.imread(img_path)

# cv2.imshow("image", img)
# cv2.waitKey(0)

# detect the faces
face_img = detect_face(imagePath="image.png")
#cv2.imshow("face", face_img)
#cv2.waitKey(0)

# percent by which the image is resized
scale_percent = 200

# calculate the 50 percent of original dimensions
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
upper_face = bw_img[0:int(np.ceil(rows / 2)), :]
lower_face = bw_img[int(np.ceil(rows / 2)):, :]
#cv2.imshow("upper", upper_face)
#cv2.imshow("lower", lower_face)
#cv2.waitKey(0)

# divide the face into left and right (our left and right, not theirs)
upper_left = upper_face[:, 0:int(np.ceil(cols / 2))]
upper_right = upper_face[:, int(np.ceil(cols / 2)):]
#cv2.imshow("upper left", upper_left)
#cv2.imshow("upper right", upper_right)
#cv2.waitKey(0)

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
    #cv2.imshow("aligned", aligned)
    bw_img = aligned

cv2.circle(copy_left, (x_left, y_left), radius=0, color=(0, 0, 255), thickness=5)
cv2.circle(copy_right, (x_right, y_right), radius=0, color=(0, 255, 0), thickness=5)
cv2.circle(face_img, (x_left, y_left), radius=0, color=(0, 0, 255), thickness=5)
cv2.circle(face_img, (x_right_real, y_right), radius=0, color=(0, 0, 255), thickness=5)

#cv2.imshow("eye l", copy_left)
#cv2.imshow("eye r", copy_right)
#cv2.imshow("eyes", bw_img)

ED = x_right_real - x_left  # eye distance

mouth_region = bw_img[y_left + int(0.85 * ED): y_left + int(1.5 * ED), x_left: x_right_real]
#cv2.imshow("mouth region", mouth_region)
#cv2.waitKey(0)

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
#cv2.imshow("mouth edges", edge)
#cv2.waitKey(0)

vert_diff_summed = np.sum(diff_ver, axis=1)

peaks, _ = find_peaks(vert_diff_summed)
results_half = peak_widths(vert_diff_summed, peaks, rel_height=0.5)
#plt.plot(np.array(range(0, np.size(vert_diff_summed))), vert_diff_summed)
#plt.hlines(*results_half[1:], color="C2")
#plt.title("Vertical Histogram of the Mouth")
#plt.xlabel("Rows")
#plt.ylabel("Pixel Intesity Differences Summed Over Columns")

#plt.show()
#plt.close()

widest_peak = peaks[np.where(results_half[0] == np.max(results_half[0]))][0]
mouth_y, mouth_x = int(widest_peak + y_left + 0.85 * ED), int((x_right_real + x_left) / 2)
#cv2.circle(face_img, (mouth_x, mouth_y), radius=0, color=(255, 0, 0), thickness=5)
#cv2.imshow("mouth ", face_img)
#cv2.waitKey(0)

all_rois = find_roi(bw_img, ED, (y_left, x_left - 10), (y_right, x_right_real - 10), (mouth_y, mouth_x - 10))

# cv2.imwrite("face 3 detected.png",face_img)

# mouthlen_div3 = int((x_right_real-x_left)/5)
# bw_copy = cv2.rectangle(bw_copy, (x_left-2*mouthlen_div3,int((y_left+0.85*ED+mouth_y)/2) ), (x_left+mouthlen_div3,int((y_left+1.5*ED+mouth_y)/2)), 255, 3)
# bw_copy = cv2.rectangle(bw_copy, (x_right_real-2*mouthlen_div3,int((y_left+0.85*ED+mouth_y)/2) ), (x_right_real+mouthlen_div3,int((y_left+1.5*ED+mouth_y)/2)), 255, 3)
# bw_copy = cv2.rectangle(bw_copy, (x_left-10,y_left-20 ), (int((x_right_real+x_left)/2)-10,y_left+20), 255, 3)

# bw_copy = cv2.rectangle(bw_copy, (int(0.85*ED),int(x_left-2*mouthlen_div3)), (int(1.5*ED+mouth_y),int(mouth_x-mouthlen_div3)), 255, 3)
# cv2.imshow("mouth rct ", bw_copy)
# cv2.waitKey(0)

# cv2.imwrite("roi detected.png",bw_copy)

filter_bank = gabor_bank()
feature_extraction(filter_bank, all_rois)
