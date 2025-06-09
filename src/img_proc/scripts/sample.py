import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('test_1.jpg')   

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower_red = (0, 100, 100)
# upper_red = (10, 255, 255)

# mask = cv2.inRange(hsv_img, lower_red, upper_red)

hist_H= cv2.calcHist([hsv_img], [0], None, [180], [0, 180])
hist_S= cv2.calcHist([hsv_img], [1], None, [256], [0, 255])
hist_V= cv2.calcHist([hsv_img], [2], None, [256], [0, 255])


max_H = np.argwhere(hist_H == np.max(hist_H))[0][0]
max_S = np.argwhere(hist_S == np.max(hist_S))[0][0]
max_V = np.argwhere(hist_V == np.max(hist_V))[0][0]

print("Max H: ", max_H)
print("Max S: ", max_S)
print("Max V: ", max_V)
MaxcolorHSV=np.uint8([[[max_H, max_S, max_V]]])
MaxcolorBGR = cv2.cvtColor(MaxcolorHSV, cv2.COLOR_HSV2BGR)[0][0]

print("MaxcolorHSV: ", MaxcolorHSV)
print("MaxcolorBGR: ", MaxcolorBGR)

if img is None:
    print("Error: Could not open or find the image.")
    sys.exit()

else:
    print("Image loaded successfully.")
    # cv2.imshow('hsv_image', hsv_img)
    cv2.imshow('Image', hsv_img)
    while True:
        key=cv2.waitKey(0)
        if key==27:
            break
cv2.destroyAllWindows()

# print(space.ndim)


