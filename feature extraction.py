import numpy as np
from math import pi, sqrt,sin,cos,ceil,floor
import cv2
def gabor_bank(size=39, num_orientation=8, num_freq=6):
    gama = 1/sqrt(pi*0.9025)
    eta = 1/sqrt(pi*0.58695)
    f_max = 0.5
    filter_bank =[]
    for o in range(num_orientation):
        for f in range(num_freq):
            f0 = f_max / ((sqrt(2)) ** (f - 1));
            teta = ((o - 1) / num_orientation) * pi;
            u = np.repeat(np.asarray(range(-floor(size/2),ceil(size/2))).reshape(1,size),size,axis=0)

            v = np.transpose(u)
            u_p = cos(teta)*u+sin(teta)*v
            v_p = -sin(teta)*u+cos(teta)*v
            c= -pi**2/f0**2*(gama*(u_p-f0)**2+eta**2*v_p**2)
            #H=np.exp(c)
            h= f0**2/pi/gama/eta*np.multiply(np.exp(-f0**2/gama**2*u_p**2-f0**2/eta**2*v_p**2),np.exp(2j*pi*f0*u_p))
            #Hh= np.fft.fft2(np.fft.fftshift(h))
            # percent by which the image is resized, just to visualize
            scale_percent = 500

            # calculate the 50 percent of original dimensions
            width = int(h.shape[1] * scale_percent / 100)
            height = int(h.shape[0] * scale_percent / 100)

            # dsize
            dsize = (width, height)

            # resize image
            h = cv2.resize(np.real(h*255/np.max(np.real(h))), dsize)
            #Hh = cv2.resize(np.real(Hh), dsize)
            #H = cv2.resize(np.real(H), dsize)
            cv2.imshow("h", h)
            #cv2.imshow("Hh", Hh)
            #cv2.imshow("H", H)
            cv2.waitKey()
            filter_bank.append(h)
    return filter_bank

gabor_bank()