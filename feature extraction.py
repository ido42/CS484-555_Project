import numpy as np
from math import pi, sqrt,sin,cos,ceil,floor
import cv2
from roi import *
def gabor_bank(size=13, num_orientation=8, num_freq=6):
    gama = 1/sqrt(pi*0.9025)
    eta = 1/sqrt(pi*0.58695)
    f_max = 0.5
    filter_bank =[]
    for o in range(num_orientation):  # iterate over different orientations
        for f in range(num_freq): # iterate over different frequencies
            f0 = f_max / ((sqrt(2)) ** (f - 1));
            teta = ((o - 1) / num_orientation) * pi;
            x = np.repeat(np.asarray(range(-floor(size/2),ceil(size/2))).reshape(1,size),size,axis=0)  # x and y are to indicate indexes
            y = np.transpose(x)
            x_p = cos(teta)*x+sin(teta)*y
            y_p = -sin(teta)*x+cos(teta)*y
            #c= -pi**2/f0**2*(gama*(u_p-f0)**2+eta**2*v_p**2)
            #H=np.exp(c)
            h= f0**2/pi/gama/eta*np.multiply(np.exp(-f0**2/gama**2*x_p**2-f0**2/eta**2*y_p**2),np.exp(2j*pi*f0*x_p))  # gabor filter in spatial domain
            #Hh= np.fft.fft2(np.fft.fftshift(h))

            # resize image, only to visualize unnecessary for real algorithm
            scale_percent = 500
            width = int(h.shape[1] * scale_percent / 100)
            height = int(h.shape[0] * scale_percent / 100)
            dsize = (width, height)
            #h = cv2.resize(np.real(h*255/np.max(np.real(h-np.min(h.real)))), dsize)

            #Hh = cv2.resize(np.real(Hh), dsize)
            #H = cv2.resize(np.real(H), dsize)
            #cv2.imshow("h", h)
            #cv2.imshow("Hh", Hh)
            #cv2.imshow("H", H)
            #cv2.waitKey()
            filter_bank.append(h)
    return filter_bank

def feature_extraction(filter_bank,ROIs):
    # at the result each row belongs to one feature, each of them are 49 linearized images first is original, rest is with filters
    roi_size =13
    num_filters =len(filter_bank)
    filtered_imgs = []
    feature_vectors =np.zeros((20,(num_filters+1)*roi_size*roi_size))
    for reg in range(len(ROIs)):
        different_filters =[]
        different_filters.append(ROIs[reg])
        orig_vect= np.reshape(ROIs[reg], (1, np.size(ROIs[reg])))  # first the original gray scale image is put to first 169 positions of the row
        feature_vectors[reg, 0:roi_size**2] = orig_vect
        for f in range(len(filter_bank)):
            # Applying the filter2D() function, the output is also 13x13 because ddepth=-1
            img = cv2.filter2D(src=ROIs[reg].astype('float64'), ddepth=-1, kernel=np.abs(filter_bank[f]))
            different_filters.append(img)
            feature_vectors[reg, (f+1)*roi_size**2:(f+2)*roi_size**2]=np.reshape(img, (1, np.size(img)))
            scale_percent = 500

            # calculate the 50 percent of original dimensions
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)

            # dsize
            dsize = (width, height)

            # resize image
            roi_filtered = cv2.resize(img/np.max(img)*255, dsize)
            #cv2.imshow("ROI:"+str(reg)+", filter:"+str(f), roi_filtered.astype('uint8'))
            print("ROI:"+str(reg)+", filter:"+str(f))
        filtered_imgs.append(different_filters)
        #cv2.waitKey()
    return feature_vectors



filter_bank = gabor_bank()
feature_extraction(filter_bank, all_rois)
