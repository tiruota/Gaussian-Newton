import numpy as np
import cv2
import math
import sys

#init parameters
theta = np.deg2rad(10)
s = 1.0

sigma = 2.0
coef = 1/(2 * np.pi*(sigma**2))
gaussian_x = [[] for _ in range(int(1+4*sigma))]
gaussian_y = [[] for _ in range(int(1+4*sigma))]

for y in range(int(1+4*sigma)):
    offset_y = round(int(4*sigma)/2) - y
    for x in range(int(1+4*sigma)):
        offset_x = x - round(int(4*sigma)/2)
        gaussian_x[y].append(-coef * (-offset_x / sigma**2) * np.exp(-1*(offset_x**2 + offset_y**2) / (2*sigma**2) ))
        gaussian_y[y].append(-coef * (-offset_y / sigma**2) * np.exp(-1*(offset_x**2 + offset_y**2) / (2*sigma**2) ))
        
gaussian_x = np.array(gaussian_x, dtype = float)
gaussian_y = np.array(gaussian_y, dtype = float)

# read images
img_in  = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
img_out = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
pixel = img_in.shape[0]

cx_in = (img_in.shape[0])/2
cy_in = cx_in
cx_out = (img_out.shape[0])/2
cy_out = cx_in

X = []
Y = []
radius = img_in.shape[0]/2
for i in range(img_in.shape[0]):
    for j in range(img_in.shape[1]):
        length = np.sqrt((i-cx_in)**2 + (j-cy_in)**2)
        if length <= radius:
            X.append(i)
            Y.append(j)

im = np.zeros(img_in.shape, dtype = float)
Ix = np.zeros(img_in.shape, dtype = float)
Iy = np.zeros(img_in.shape, dtype = float)

ite = 0
init_error = 0
while(1):
    ite += 1
    J_0 = 0
    J_00 = 0
    J_s = 0
    J_ss = 0
    J_0s = 0
    error = 0

    #rot(θ)
    for j in range(len(X)):
        x = int(X[j])
        y = int(Y[j])
        x_rot = int(round( ((x-cx_in) * np.cos(-theta) - (y-cy_in) * np.sin(-theta) ) + cx_in))
        y_rot = int(round( ((x-cx_in) * np.sin(-theta) + (y-cy_in) * np.cos(-theta) ) + cy_in))
        if x_rot >= pixel or y_rot >= pixel:
            im[y, x] = 0
        else:
            im[y, x] = img_out[y_rot, x_rot]
            
    # smooth differential
    im = im.astype(np.float64)
    Ixt = cv2.filter2D(im, -1, gaussian_x)
    Iyt = cv2.filter2D(im, -1, gaussian_y)

    #rot(-θ)
    for j in range(len(X)):
        x = X[j]
        y = Y[j]
        x_rot = int(round( ((x-cx_in) * np.cos(theta) - (y-cy_in) * np.sin(theta)) + cx_in))
        y_rot = int(round( ((x-cx_in) * np.sin(theta) + (y-cy_in) * np.cos(theta)) + cy_in))
        if x_rot >= pixel or y_rot >= pixel:
            Ix[y, x] = 0
            Iy[y, x] = 0
        else:
            Ix[y, x] = Ixt[y_rot, x_rot]
            Iy[y, x] = Iyt[y_rot, x_rot]

    for j in range(len(X)):
        x = X[j]
        y = Y[j]

        diff_x = int(round( s *  ( (x - cx_in) * np.cos(theta) - (cy_in-y) * np.sin(theta) ) +cx_in))
        diff_y = int(round( -s * ( (x - cx_in) * np.sin(theta) + (cy_in-y) * np.cos(theta) ) +cy_in))

        if diff_x >= pixel or diff_y >= pixel or diff_x < 0 or diff_y < 0:
            diff_I = 0
            im[y,x] = 0
            diff_Ix = 0
            diff_Iy = 0
        else:
            diff_I = img_out[diff_y, diff_x]
            im[y,x] = img_out[diff_y, diff_x]
            diff_Ix = Ix[diff_y, diff_x]
            diff_Iy = Iy[diff_y, diff_x]
            diff_I = diff_I.astype('int64')
        I = img_in[y,x]

        dx_theta = (s* (-(x-cx_in) * np.sin(theta) - (cy_in-y) * np.cos(theta)))
        dy_theta = (s* ( (x-cx_in) * np.cos(theta) - (cy_in-y) * np.sin(theta)))

        dx_s = ((x-cx_in) * np.cos(theta) - (cy_in-y) * np.sin(theta))
        dy_s = ((x-cx_in) * np.sin(theta) + (cy_in-y) * np.cos(theta))
        
        dx_theta_s = (-(x-cx_in) * np.sin(theta) - (cy_in-y) * np.cos(theta))
        dy_theta_s = ( (x-cx_in) * np.cos(theta) - (cy_in-y) * np.sin(theta))

        I = I.astype('int64')
        
        J_0 += (diff_I - I) * (diff_Ix * dx_theta + diff_Iy * dy_theta)
        J_00 += (diff_Ix * dx_theta + diff_Iy * dy_theta)**2
        J_s += (diff_I - I) * (diff_Ix * dx_s + diff_Iy * dy_s)
        J_ss += (diff_Ix * dx_s + diff_Iy * dy_s)**2                 
        J_0s += diff_Ix**2 * dx_theta * dx_s + diff_Ix * diff_Ix * ( dx_theta * dy_s + dx_s * dy_theta) + diff_Iy**2 * dy_theta * dy_s

        error = error + (1/2) * (diff_I - I)**2
    
    J = np.array([[J_00, J_0s],
                [J_0s, J_ss]])
    J_vec = np.array([[J_0],[J_s]])

    [dtheta, ds] = np.matmul(np.linalg.inv(J), J_vec)
    theta = theta - dtheta[0]
    s = s - ds[0]
    
    print("iterate=%d theta=%f s=%f error=%f"%(ite, np.rad2deg(theta), s, error))
    if init_error == 0:
            init_error = error
    if error < init_error * 0.01:
        break
