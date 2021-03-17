import numpy as np
import cv2

#get rgb values from here: https://imagecolorpicker.com/en
rgbColor = np.uint8([[[169,169,169]]])
hsvColor = cv2.cvtColor(rgbColor, cv2.COLOR_BGR2HSV)

upper = np.array([hsvColor[0][0][0] + 10, hsvColor[0][0][1] + 10, hsvColor[0][0][2] + 40])
lower = np.array([hsvColor[0][0][0] - 10, hsvColor[0][0][1] - 10, hsvColor[0][0][2] - 40])

print(lower)
print(np.array([hsvColor[0][0][0], hsvColor[0][0][1], hsvColor[0][0][2]]))
print(upper)